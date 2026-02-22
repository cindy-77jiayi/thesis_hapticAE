"""Plotting utilities for training and evaluation."""

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    title: str = "Training Loss",
    save_path: str | None = None,
):
    """Plot train/val loss curves."""
    plt.figure(figsize=(8, 3))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_waveform_comparison(
    x_np: np.ndarray,
    xhat_np: np.ndarray,
    n_show: int | None = None,
    title_prefix: str = "Sample",
    save_path: str | None = None,
):
    """Plot original vs reconstructed waveforms."""
    n = n_show or len(x_np)
    n = min(n, len(x_np))

    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))
    if n == 1:
        axes = [axes]

    for i in range(n):
        axes[i].plot(x_np[i], label="original")
        axes[i].plot(xhat_np[i], label="reconstructed", alpha=0.8)
        orig_max = np.max(np.abs(x_np[i]))
        recon_max = np.max(np.abs(xhat_np[i]))
        axes[i].set_title(f"{title_prefix} {i} | orig max={orig_max:.3f}, recon max={recon_max:.3f}")
        axes[i].legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
