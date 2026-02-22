"""Unified trainer for VAE and AE models."""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import (
    amplitude_loss,
    fft_mag_mse,
    kl_divergence_free_bits,
    multi_scale_spectral_loss,
)
from .schedulers import cyclical_beta_schedule


class Trainer:
    """Config-driven trainer supporting both VAE and AE models.

    Handles training loop, validation, early stopping, checkpointing,
    and metric logging.
    """

    def __init__(self, model, config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.is_vae = config.get("model_type", "vae") == "vae"

        # Training params
        train_cfg = config.get("training", {})
        self.total_epochs = train_cfg.get("epochs", 100)
        self.patience = train_cfg.get("patience", 15)
        self.min_delta = train_cfg.get("min_delta", 1e-4)
        self.early_stop_start = train_cfg.get("early_stop_start", 10)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.print_every = train_cfg.get("print_every", 10)

        # Loss weights
        loss_cfg = config.get("loss", {})
        self.w_l1 = loss_cfg.get("l1_weight", 0.2)
        self.w_spec = loss_cfg.get("spectral_weight", 0.15)
        self.w_amp = loss_cfg.get("amplitude_weight", 0.5)
        self.w_fft = loss_cfg.get("fft_weight", 0.0)
        self.clamp_range = loss_cfg.get("clamp_range", 3.0)

        # KL params (VAE only)
        kl_cfg = config.get("kl", {})
        self.free_bits = kl_cfg.get("free_bits", 0.1)
        self.beta_max = kl_cfg.get("beta_max", 0.0001)
        self.n_cycles = kl_cfg.get("n_cycles", 4)
        self.beta_ratio = kl_cfg.get("ratio", 0.5)

        # Optimizer
        opt_cfg = config.get("optimizer", {})
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 2e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-5),
        )

        sched_cfg = config.get("scheduler", {})
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 15),
        )

        # Output directory
        output_dir = config.get("output_dir", "outputs")
        run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.run_dir = os.path.join(output_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        self.ckpt_path = os.path.join(self.run_dir, "best_model.pt")
        self.metrics_path = os.path.join(self.run_dir, "metrics.npz")

        # State
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val = float("inf")
        self.best_state = None

    def _compute_loss(self, x, epoch):
        """Forward pass + loss computation for both VAE and AE."""
        if self.is_vae:
            x_hat_raw, mu, logvar, z = self.model(x)
        else:
            x_hat_raw, z = self.model(x)
            mu = logvar = None

        x_hat = torch.clamp(x_hat_raw, -self.clamp_range, self.clamp_range)

        mse = torch.nn.functional.mse_loss(x_hat, x)
        l1 = torch.nn.functional.l1_loss(x_hat, x)
        recon = mse + self.w_l1 * l1

        if self.w_spec > 0:
            recon = recon + self.w_spec * multi_scale_spectral_loss(x_hat, x)
        if self.w_amp > 0:
            recon = recon + self.w_amp * amplitude_loss(x_hat, x)
        if self.w_fft > 0:
            recon = recon + self.w_fft * fft_mag_mse(x_hat, x)

        loss = recon
        kl_val = 0.0

        if self.is_vae and mu is not None:
            kl = kl_divergence_free_bits(mu, logvar, free_bits=self.free_bits)
            beta = cyclical_beta_schedule(
                epoch, self.total_epochs,
                n_cycles=self.n_cycles,
                ratio=self.beta_ratio,
                beta_max=self.beta_max,
            )
            loss = recon + beta * kl
            kl_val = kl.detach().item()

        return loss, recon.detach().item(), kl_val, x_hat_raw

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> float:
        self.model.train(train)
        total_loss = 0.0
        count = 0
        mode = "train" if train else "val"

        pbar = tqdm(loader, leave=False, desc=f"{mode} epoch {epoch}")
        for batch_idx, x in enumerate(pbar):
            x = x.to(self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            loss, recon, kl_val, x_hat_raw = self._compute_loss(x, epoch)

            if not torch.isfinite(loss):
                continue

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            bs = x.shape[0]
            total_loss += loss.detach().item() * bs
            count += bs

            pbar.set_postfix(loss=f"{total_loss / max(count, 1):.4f}")

        return total_loss / max(count, 1)

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Full training loop with early stopping and checkpointing.

        Returns a dict with train_losses, val_losses, best_val.
        """
        wait = 0

        for epoch in range(1, self.total_epochs + 1):
            tr = self._run_epoch(train_loader, train=True, epoch=epoch)
            va = self._run_epoch(val_loader, train=False, epoch=epoch)

            self.train_losses.append(tr)
            self.val_losses.append(va)

            if np.isfinite(va):
                self.scheduler.step(va)

            lr = self.optimizer.param_groups[0]["lr"]
            if epoch % self.print_every == 0 or epoch == 1:
                pct = 100.0 * epoch / self.total_epochs
                print(
                    f"[Epoch {epoch:03d}/{self.total_epochs} | {pct:5.1f}%] "
                    f"lr={lr:.2e} | train={tr:.6f} | val={va:.6f}"
                )

            if epoch >= self.early_stop_start:
                if not np.isfinite(va):
                    wait += 1
                elif va < self.best_val - self.min_delta:
                    self.best_val = va
                    self.best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
                    wait = 0
                    torch.save(self.best_state, self.ckpt_path)
                else:
                    wait += 1
                    if wait >= self.patience:
                        print(f"⏹️ Early stopping at epoch {epoch}, best val = {self.best_val:.6f}")
                        break

        # Load best and save metrics
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.model.to(self.device)
        self.model.eval()

        np.savez(
            self.metrics_path,
            train_losses=np.array(self.train_losses),
            val_losses=np.array(self.val_losses),
            best_val=self.best_val,
        )

        print(f"✅ Best val: {self.best_val:.6f}")
        print(f"📁 Checkpoint: {self.ckpt_path}")
        print(f"📊 Metrics: {self.metrics_path}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val": self.best_val,
            "run_dir": self.run_dir,
        }
