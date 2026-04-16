"""Trainer for the control branch built on top of a frozen codec."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.signal_metrics import compute_all_metrics

from .losses import amplitude_loss, envelope_loss


def compute_metric_targets(x: torch.Tensor, metric_names: list[str], sr: int) -> torch.Tensor:
    """Compute selected haptic descriptors for a batch."""
    arr = x.detach().cpu().numpy()[:, 0, :]
    rows = []
    for sample in arr:
        metrics = compute_all_metrics(sample, sr=sr)
        rows.append([float(metrics[name]) for name in metric_names])
    return torch.tensor(rows, dtype=torch.float32)


class ControlTrainer:
    """Train the control encoder/decoder and descriptor head."""

    def __init__(self, model, config: dict, device: torch.device, metric_stats: dict[str, list[float]]):
        self.model = model.to(device)
        self.model.freeze_codec()
        self.config = config
        self.device = device
        self.metric_names = list(config["control_loss"]["metric_names"])
        self.metric_mean = torch.tensor(metric_stats["mean"], dtype=torch.float32, device=device)
        self.metric_std = torch.tensor(metric_stats["std"], dtype=torch.float32, device=device)
        self.sample_rate = int(config["data"]["sr"])

        train_cfg = config.get("control_training", {})
        self.total_epochs = int(train_cfg.get("epochs", 40))
        self.patience = int(train_cfg.get("patience", 8))
        self.min_delta = float(train_cfg.get("min_delta", 1e-4))
        self.early_stop_start = int(train_cfg.get("early_stop_start", 6))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.print_every = int(train_cfg.get("print_every", 5))

        loss_cfg = config.get("control_loss", {})
        self.w_l1 = float(loss_cfg.get("waveform_l1_weight", 0.3))
        self.w_amp = float(loss_cfg.get("amplitude_weight", 1.0))
        self.w_env = float(loss_cfg.get("envelope_weight", 1.0))
        self.w_metric = float(loss_cfg.get("metric_weight", 1.0))
        self.clamp_range = float(loss_cfg.get("clamp_range", 3.0))

        opt_cfg = config.get("control_optimizer", {})
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            params,
            lr=float(opt_cfg.get("lr", 2e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
        )
        sched_cfg = config.get("control_scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 6)),
        )

        output_dir = config.get("output_dir", "outputs")
        run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.run_dir, "best_control.pt")
        self.metrics_path = os.path.join(self.run_dir, "control_metrics.npz")
        self.summary_path = os.path.join(self.run_dir, "control_summary.json")

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val = float("inf")
        self.best_state = None

    def _compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        z_ctrl = self.model.encode_control(x)
        recon_ctrl = torch.clamp(self.model.decode_control(z_ctrl, target_len=x.shape[-1]), -self.clamp_range, self.clamp_range)
        pred_metrics = self.model.predict_metrics(z_ctrl)

        target_metrics = compute_metric_targets(x, self.metric_names, self.sample_rate).to(self.device)
        target_norm = (target_metrics - self.metric_mean) / self.metric_std.clamp_min(1e-6)

        l1 = torch.nn.functional.l1_loss(recon_ctrl, x)
        amp = amplitude_loss(recon_ctrl, x)
        env = envelope_loss(recon_ctrl, x, sr=self.sample_rate)
        metric = torch.nn.functional.mse_loss(pred_metrics, target_norm)

        loss = self.w_l1 * l1 + self.w_amp * amp + self.w_env * env + self.w_metric * metric
        stats = {
            "l1": float(l1.detach().item()),
            "amp": float(amp.detach().item()),
            "env": float(env.detach().item()),
            "metric": float(metric.detach().item()),
        }
        return loss, stats

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train(train)
        total_loss = 0.0
        count = 0
        stats_accum = {"l1": 0.0, "amp": 0.0, "env": 0.0, "metric": 0.0}
        mode = "train" if train else "val"
        pbar = tqdm(loader, desc=f"{mode} control {epoch}", leave=False)

        for x in pbar:
            x = x.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._compute_loss(x)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
                self.optimizer.step()

            batch = x.shape[0]
            total_loss += float(loss.detach().item()) * batch
            count += batch
            for key, val in stats.items():
                stats_accum[key] += val * batch
            pbar.set_postfix(loss=f"{total_loss / max(count, 1):.4f}")

        return total_loss / max(count, 1), {key: val / max(count, 1) for key, val in stats_accum.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        wait = 0
        last_train_stats: dict[str, float] = {}
        last_val_stats: dict[str, float] = {}
        for epoch in range(1, self.total_epochs + 1):
            tr, tr_stats = self._run_epoch(train_loader, True, epoch)
            va, va_stats = self._run_epoch(val_loader, False, epoch)
            self.train_losses.append(tr)
            self.val_losses.append(va)
            last_train_stats = tr_stats
            last_val_stats = va_stats
            self.scheduler.step(va)

            if epoch % self.print_every == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                pct = 100.0 * epoch / self.total_epochs
                print(
                    f"[Control {epoch:03d}/{self.total_epochs} | {pct:5.1f}%] "
                    f"lr={lr:.2e} | train={tr:.6f} | val={va:.6f} | metric={va_stats['metric']:.5f}"
                )

            improved = np.isfinite(va) and va < self.best_val - self.min_delta
            if improved:
                self.best_val = va
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_state, self.ckpt_path)
                wait = 0
            elif epoch >= self.early_stop_start:
                wait += 1
                if wait >= self.patience:
                    print(f"⏹️ Early stopping at epoch {epoch}, best val = {self.best_val:.6f}")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        np.savez(
            self.metrics_path,
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses),
            best_val=self.best_val,
        )
        with open(self.summary_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "best_val": self.best_val,
                    "metric_names": self.metric_names,
                    "train_components_last": last_train_stats,
                    "val_components_last": last_val_stats,
                },
                fp,
                indent=2,
            )
        print(f"✅ Best control val: {self.best_val:.6f}")
        print(f"📁 Checkpoint: {self.ckpt_path}")
        return {
            "run_dir": self.run_dir,
            "best_val": self.best_val,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
