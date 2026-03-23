"""Quantitative evaluation of reconstruction quality."""

import numpy as np
import torch
from torch.utils.data import DataLoader


def evaluate_reconstruction(
    model,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 10,
    is_vae: bool = True,
) -> dict:
    """Run model on a batch from the loader and compute quality metrics.

    Returns dict with original/reconstructed arrays, latent stats, and per-sample metrics.
    """
    model.eval()
    x = next(iter(loader))[:n_samples].to(device)

    with torch.no_grad():
        if is_vae:
            x_hat, mu, logvar, z = model(x)
        else:
            x_hat, z = model(x)
            mu = logvar = None

    x_np = x[:, 0, :].cpu().numpy()
    xhat_np = x_hat[:, 0, :].cpu().numpy()

    metrics = []
    for i in range(len(x_np)):
        orig_std = np.std(x_np[i])
        recon_std = np.std(xhat_np[i])
        orig_max = np.max(np.abs(x_np[i]))
        recon_max = np.max(np.abs(xhat_np[i]))
        ratio = recon_std / (orig_std + 1e-8)
        metrics.append({
            "sample": i,
            "orig_std": orig_std,
            "recon_std": recon_std,
            "std_ratio": ratio,
            "orig_max": orig_max,
            "recon_max": recon_max,
        })

    result = {
        "x_np": x_np,
        "xhat_np": xhat_np,
        "z": z.cpu().numpy(),
        "per_sample": metrics,
    }

    if mu is not None:
        result["mu_mean"] = mu.mean().item()
        result["mu_std"] = mu.std().item()
        result["logvar_mean"] = logvar.mean().item()

    return result


def print_metrics(result: dict):
    """Pretty-print reconstruction quality metrics."""
    print("Reconstruction Quality (Standard Deviation):")
    print("-" * 70)
    for m in result["per_sample"]:
        print(
            f"  Sample {m['sample']:2d}: "
            f"Orig STD={m['orig_std']:.4f}  Recon STD={m['recon_std']:.4f}  "
            f"Ratio={m['std_ratio']:.2%}  "
            f"Orig Max={m['orig_max']:.4f}  Recon Max={m['recon_max']:.4f}"
        )
    print("-" * 70)
    if "mu_mean" in result:
        print(
            f"  Latent: mu mean={result['mu_mean']:.4f}  "
            f"mu std={result['mu_std']:.4f}  "
            f"logvar mean={result['logvar_mean']:.4f}"
        )
