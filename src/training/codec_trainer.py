"""Trainer for the sequence-codec reconstruction stage."""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import amplitude_loss, envelope_loss, multiscale_stft_loss


class CodecTrainer:
    """Train the reconstruction codec branch and track best checkpoints."""

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.total_epochs = int(train_cfg.get("epochs", 50))
        self.patience = int(train_cfg.get("patience", 10))
        self.min_delta = float(train_cfg.get("min_delta", 1e-4))
        self.early_stop_start = int(train_cfg.get("early_stop_start", 8))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.print_every = int(train_cfg.get("print_every", 5))

        loss_cfg = config.get("loss", {})
        self.w_l1 = float(loss_cfg.get("l1_weight", 0.4))
        self.w_stft = float(loss_cfg.get("stft_loss_weight", 0.5))
        self.w_amp = float(loss_cfg.get("amplitude_weight", 1.0))
        self.w_env = float(loss_cfg.get("envelope_weight", 1.0))
        self.recon_time_weight = float(loss_cfg.get("recon_time_weight", 1.0))
        self.clamp_range = float(loss_cfg.get("clamp_range", 3.0))
        self.stft_scales = loss_cfg.get("stft_scales", [128, 256, 512, 1024])
        self.stft_hop_lengths = loss_cfg.get("stft_hop_lengths")
        self.stft_win_lengths = loss_cfg.get("stft_win_lengths")
        self.stft_scale_weights = loss_cfg.get("stft_scale_weights")
        self.stft_linear_weight = float(loss_cfg.get("stft_linear_weight", 0.02))
        self.stft_log_weight = float(loss_cfg.get("stft_log_weight", 0.02))
        self.stft_eps = float(loss_cfg.get("stft_eps", 1e-7))
        self.sample_rate = int(config["data"]["sr"])

        opt_cfg = config.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(opt_cfg.get("lr", 2e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
        )

        sched_cfg = config.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 8)),
        )

        output_dir = config.get("output_dir", "outputs")
        run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.run_dir, "best_model.pt")
        self.metrics_path = os.path.join(self.run_dir, "metrics.npz")
        self.summary_path = os.path.join(self.run_dir, "training_summary.json")

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val = float("inf")
        self.best_state = None

    def _compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        recon_seq, _, _, _, _, vq_losses = self.model(x)
        recon_seq = torch.clamp(recon_seq, -self.clamp_range, self.clamp_range)

        mse = torch.nn.functional.mse_loss(recon_seq, x)
        l1 = torch.nn.functional.l1_loss(recon_seq, x)
        time_loss = self.recon_time_weight * (mse + self.w_l1 * l1)

        stft = multiscale_stft_loss(
            recon_seq,
            x,
            stft_scales=self.stft_scales,
            stft_hop_lengths=self.stft_hop_lengths,
            stft_win_lengths=self.stft_win_lengths,
            stft_scale_weights=self.stft_scale_weights,
            stft_linear_weight=self.stft_linear_weight,
            stft_log_weight=self.stft_log_weight,
            eps=self.stft_eps,
        ) if self.w_stft > 0 else torch.tensor(0.0, device=x.device)

        amp = amplitude_loss(recon_seq, x) if self.w_amp > 0 else torch.tensor(0.0, device=x.device)
        env = envelope_loss(recon_seq, x, sr=self.sample_rate) if self.w_env > 0 else torch.tensor(0.0, device=x.device)
        vq = vq_losses["vq_loss"]

        loss = time_loss + self.w_stft * stft + self.w_amp * amp + self.w_env * env + vq
        stats = {
            "mse": float(mse.detach().item()),
            "l1": float(l1.detach().item()),
            "stft": float(stft.detach().item()),
            "amp": float(amp.detach().item()),
            "env": float(env.detach().item()),
            "vq": float(vq.detach().item()),
        }
        return loss, stats

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> tuple[float, dict[str, float]]:
        self.model.train(train)
        total_loss = 0.0
        count = 0
        stats_accum: dict[str, float] = {"mse": 0.0, "l1": 0.0, "stft": 0.0, "amp": 0.0, "env": 0.0, "vq": 0.0}
        mode = "train" if train else "val"
        pbar = tqdm(loader, desc=f"{mode} epoch {epoch}", leave=False)

        for x in pbar:
            x = x.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            loss, stats = self._compute_loss(x)
            if not torch.isfinite(loss):
                continue
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            batch = x.shape[0]
            total_loss += float(loss.detach().item()) * batch
            count += batch
            for key, val in stats.items():
                stats_accum[key] += val * batch
            pbar.set_postfix(loss=f"{total_loss / max(count, 1):.4f}")

        mean_stats = {key: val / max(count, 1) for key, val in stats_accum.items()}
        return total_loss / max(count, 1), mean_stats

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

            if np.isfinite(va):
                self.scheduler.step(va)

            if epoch % self.print_every == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                pct = 100.0 * epoch / self.total_epochs
                print(
                    f"[Epoch {epoch:03d}/{self.total_epochs} | {pct:5.1f}%] "
                    f"lr={lr:.2e} | train={tr:.6f} | val={va:.6f} | "
                    f"vq={va_stats['vq']:.5f}"
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
                    "train_loss_last": self.train_losses[-1] if self.train_losses else None,
                    "val_loss_last": self.val_losses[-1] if self.val_losses else None,
                    "train_components_last": last_train_stats,
                    "val_components_last": last_val_stats,
                },
                fp,
                indent=2,
            )

        print(f"✅ Best val: {self.best_val:.6f}")
        print(f"📁 Checkpoint: {self.ckpt_path}")
        return {
            "run_dir": self.run_dir,
            "best_val": self.best_val,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
