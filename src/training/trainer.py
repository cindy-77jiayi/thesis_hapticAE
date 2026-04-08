"""Unified trainer for VAE and AE models with resumable checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.visualize import plot_waveform_comparison

from .builders import build_ema, build_optimizer, build_scheduler
from .checkpointing import CheckpointManager, RunArtifacts, save_history_json
from .losses import (
    amplitude_loss,
    fft_mag_mse,
    kl_divergence_free_bits,
    multiscale_stft_loss,
    multi_scale_spectral_loss,
)
from .schedulers import cyclical_beta_schedule


class Trainer:
    """Config-driven trainer supporting both VAE and AE models."""

    def __init__(
        self,
        model,
        config: dict,
        device: torch.device,
        artifacts: RunArtifacts,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.artifacts = artifacts
        self.is_vae = config.get("model_type", "vae") == "vae"

        train_cfg = config.get("training", {})
        self.total_epochs = train_cfg.get("epochs", 100)
        self.patience = train_cfg.get("patience", 15)
        self.min_delta = train_cfg.get("min_delta", 1e-4)
        self.early_stop_start = train_cfg.get("early_stop_start", 10)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.print_every = train_cfg.get("print_every", 10)

        loss_cfg = config.get("loss", {})
        self.w_l1 = loss_cfg.get("l1_weight", 0.2)
        self.w_spec = loss_cfg.get("spectral_weight", 0.15)
        self.w_amp = loss_cfg.get("amplitude_weight", 0.5)
        self.w_fft = loss_cfg.get("fft_weight", 0.0)
        self.clamp_range = loss_cfg.get("clamp_range", 3.0)
        self.recon_time_weight = loss_cfg.get("recon_time_weight", 1.0)

        self.use_multiscale_stft_loss = loss_cfg.get("use_multiscale_stft_loss", False)
        self.stft_scales = loss_cfg.get("stft_scales", [128, 256, 512, 1024])
        self.stft_hop_lengths = loss_cfg.get("stft_hop_lengths", None)
        self.stft_win_lengths = loss_cfg.get("stft_win_lengths", None)
        self.stft_scale_weights = loss_cfg.get("stft_scale_weights", None)
        self.stft_linear_weight = loss_cfg.get("stft_linear_weight", 0.1)
        self.stft_log_weight = loss_cfg.get("stft_log_weight", 0.1)
        self.stft_eps = loss_cfg.get("stft_eps", 1e-7)

        kl_cfg = config.get("kl", {})
        self.free_bits = kl_cfg.get("free_bits", 0.1)
        self.beta_max = kl_cfg.get("beta_max", 0.0001)
        self.n_cycles = kl_cfg.get("n_cycles", 4)
        self.beta_ratio = kl_cfg.get("ratio", 0.5)

        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        self.ema = build_ema(self.model, config)
        self.checkpoints = CheckpointManager(artifacts, config)

        val_cfg = config.get("validation", {})
        self.sample_every = int(val_cfg.get("sample_every", 0) or 0)
        self.sample_n = int(val_cfg.get("n_samples", 4))

        self.history: list[dict] = []
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.start_epoch = 1
        self.wait = 0

    def _compute_loss(self, x, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Forward pass + loss computation for both VAE and AE."""
        if self.is_vae:
            x_hat_raw, mu, logvar, z = self.model(x)
        else:
            x_hat_raw, z = self.model(x)
            mu = logvar = None

        x_hat = torch.clamp(x_hat_raw, -self.clamp_range, self.clamp_range)

        mse = torch.nn.functional.mse_loss(x_hat, x)
        l1 = torch.nn.functional.l1_loss(x_hat, x)
        time_recon = mse + self.w_l1 * l1
        recon = self.recon_time_weight * time_recon

        spectral = torch.zeros((), device=x.device)
        if self.use_multiscale_stft_loss:
            spectral = multiscale_stft_loss(
                x_hat,
                x,
                stft_scales=self.stft_scales,
                stft_hop_lengths=self.stft_hop_lengths,
                stft_win_lengths=self.stft_win_lengths,
                stft_scale_weights=self.stft_scale_weights,
                stft_linear_weight=self.stft_linear_weight,
                stft_log_weight=self.stft_log_weight,
                eps=self.stft_eps,
            )
            recon = recon + spectral
        elif self.w_spec > 0:
            spectral = self.w_spec * multi_scale_spectral_loss(x_hat, x)
            recon = recon + spectral

        amp = torch.zeros((), device=x.device)
        if self.w_amp > 0:
            amp = self.w_amp * amplitude_loss(x_hat, x)
            recon = recon + amp

        fft = torch.zeros((), device=x.device)
        if self.w_fft > 0:
            fft = self.w_fft * fft_mag_mse(x_hat, x)
            recon = recon + fft

        loss = recon
        beta = 0.0
        kl_value = torch.zeros((), device=x.device)
        if self.is_vae and mu is not None:
            kl_value = kl_divergence_free_bits(mu, logvar, free_bits=self.free_bits)
            beta = cyclical_beta_schedule(
                epoch,
                self.total_epochs,
                n_cycles=self.n_cycles,
                ratio=self.beta_ratio,
                beta_max=self.beta_max,
            )
            loss = recon + beta * kl_value

        metrics = {
            "loss": float(loss.detach().item()),
            "mse": float(mse.detach().item()),
            "l1": float(l1.detach().item()),
            "recon": float(recon.detach().item()),
            "spectral": float(spectral.detach().item()),
            "amplitude": float(amp.detach().item()),
            "fft": float(fft.detach().item()),
            "kl": float(kl_value.detach().item()),
            "beta": float(beta),
        }
        return loss, metrics

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> dict[str, float]:
        self.model.train(train)
        mode = "train" if train else "val"
        metric_totals: dict[str, float] = {}
        count = 0

        pbar = tqdm(loader, leave=False, desc=f"{mode} epoch {epoch}")
        for x in pbar:
            x = x.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            loss, metrics = self._compute_loss(x, epoch)
            if not torch.isfinite(loss):
                continue

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            bs = x.shape[0]
            count += bs
            for key, value in metrics.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value * bs

            avg_loss = metric_totals.get("loss", 0.0) / max(count, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        if count == 0:
            return {"loss": float("inf")}
        return {key: value / count for key, value in metric_totals.items()}

    def _collect_model_state(self, use_ema: bool = False) -> dict:
        if use_ema and self.ema is not None:
            return self.ema.averaged_state_dict()
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _save_validation_preview(self, loader: DataLoader, epoch: int) -> None:
        if self.sample_every <= 0 or epoch % self.sample_every != 0:
            return

        batch = next(iter(loader))[: self.sample_n].to(self.device)
        preview_context = self.ema.apply_to(self.model) if self.ema is not None else nullcontext()
        with torch.no_grad(), preview_context:
            if self.is_vae:
                x_hat, _, _, _ = self.model(batch)
            else:
                x_hat, _ = self.model(batch)
            x_hat = torch.clamp(x_hat, -self.clamp_range, self.clamp_range)

        x_np = batch[:, 0, :].detach().cpu().numpy()
        xhat_np = x_hat[:, 0, :].detach().cpu().numpy()
        epoch_dir = os.path.join(self.artifacts.preview_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        np.save(os.path.join(epoch_dir, "original.npy"), x_np)
        np.save(os.path.join(epoch_dir, "reconstructed.npy"), xhat_np)
        plot_waveform_comparison(
            x_np,
            xhat_np,
            n_show=min(len(x_np), self.sample_n),
            title_prefix=f"Epoch {epoch}",
            save_path=os.path.join(epoch_dir, "waveforms.png"),
        )

        preview_summary = {
            "epoch": epoch,
            "n_samples": int(len(x_np)),
            "mse_mean": float(np.mean((xhat_np - x_np) ** 2)),
            "mae_mean": float(np.mean(np.abs(xhat_np - x_np))),
        }
        with open(os.path.join(epoch_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(preview_summary, f, indent=2)

    def _save_metrics(self) -> None:
        np.savez(
            self.artifacts.metrics_path,
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses),
            best_val=self.best_val,
        )
        save_history_json(self.artifacts.history_path, self.history)

    def _training_state(self, epoch: int) -> dict:
        return {
            "format_version": 2,
            "epoch": epoch,
            "model_state": self._collect_model_state(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_state": self.ema.state_dict() if self.ema is not None else None,
            "best_state": self.best_state,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "wait": self.wait,
            "history": self.history,
            "config": self.config,
        }

    def restore(self, checkpoint_path: str) -> None:
        """Restore the trainer/model/optimizer state from a saved checkpoint."""
        state = self.checkpoints.load_training_checkpoint(checkpoint_path)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        if self.ema is not None and state.get("ema_state") is not None:
            self.ema.load_state_dict(state["ema_state"])

        self.best_state = state.get("best_state")
        self.best_val = float(state.get("best_val", float("inf")))
        self.best_epoch = int(state.get("best_epoch", 0))
        self.wait = int(state.get("wait", 0))
        self.history = list(state.get("history", []))
        self.train_losses = [float(item["train"]["loss"]) for item in self.history if "train" in item]
        self.val_losses = [float(item["val"]["loss"]) for item in self.history if "val" in item]
        self.start_epoch = int(state["epoch"]) + 1

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop with early stopping, EMA, and resumable checkpoints."""
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_context = self.ema.apply_to(self.model) if self.ema is not None else nullcontext()
            with val_context:
                val_metrics = self._run_epoch(val_loader, train=False, epoch=epoch)

            self.train_losses.append(float(train_metrics["loss"]))
            self.val_losses.append(float(val_metrics["loss"]))
            self.history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            })

            if np.isfinite(val_metrics["loss"]):
                self.scheduler.step(val_metrics["loss"])

            lr = self.optimizer.param_groups[0]["lr"]
            if epoch % self.print_every == 0 or epoch == 1:
                pct = 100.0 * epoch / self.total_epochs
                beta = train_metrics.get("beta", 0.0)
                print(
                    f"[Epoch {epoch:03d}/{self.total_epochs} | {pct:5.1f}%] "
                    f"lr={lr:.2e} beta={beta:.4f} "
                    f"train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f}"
                )

            improved = False
            if np.isfinite(val_metrics["loss"]) and val_metrics["loss"] < self.best_val - self.min_delta:
                self.best_val = float(val_metrics["loss"])
                self.best_epoch = epoch
                self.best_state = self._collect_model_state(use_ema=self.ema is not None)
                self.checkpoints.save_best_model(self.best_state)
                improved = True

            if epoch >= self.early_stop_start:
                if improved:
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        print(f"Early stopping at epoch {epoch}, best val = {self.best_val:.6f}")
                        self._save_metrics()
                        self.checkpoints.save_training_checkpoint(self._training_state(epoch), epoch)
                        break

            self._save_validation_preview(val_loader, epoch)
            self._save_metrics()
            self.checkpoints.save_training_checkpoint(self._training_state(epoch), epoch)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        self.model.eval()

        print(f"Best val: {self.best_val:.6f} at epoch {self.best_epoch}")
        print(f"Best checkpoint: {self.artifacts.best_model_path}")
        print(f"Resume checkpoint: {self.artifacts.last_checkpoint_path}")
        print(f"Metrics: {self.artifacts.metrics_path}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val": self.best_val,
            "best_epoch": self.best_epoch,
            "run_dir": self.artifacts.run_dir,
        }

