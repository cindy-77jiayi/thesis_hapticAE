"""Quantitative evaluation of reconstruction quality."""

from pathlib import Path

import numpy as np
import torch
from scipy.signal import hilbert
from torch.utils.data import DataLoader


def _si_snr_db(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant signal-to-noise ratio in dB."""
    ref_zm = ref - np.mean(ref)
    est_zm = est - np.mean(est)
    ref_energy = np.sum(ref_zm ** 2) + eps
    proj = np.sum(est_zm * ref_zm) * ref_zm / ref_energy
    noise = est_zm - proj
    return float(10.0 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def _lsd_db(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    """Log-spectral distance (dB) using rFFT magnitude."""
    ref_mag = np.abs(np.fft.rfft(ref))
    est_mag = np.abs(np.fft.rfft(est))
    diff = 20.0 * np.log10(ref_mag + eps) - 20.0 * np.log10(est_mag + eps)
    return float(np.sqrt(np.mean(diff ** 2)))


def _envelope_corr(ref: np.ndarray, est: np.ndarray) -> float:
    """Pearson correlation between analytic envelopes."""
    env_ref = np.abs(hilbert(ref))
    env_est = np.abs(hilbert(est))
    if np.std(env_ref) < 1e-10 or np.std(env_est) < 1e-10:
        return 0.0
    return float(np.corrcoef(env_ref, env_est)[0, 1])


def evaluate_reconstruction(
    model,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 10,
    is_vae: bool = True,
) -> dict:
    """Run model on a batch from the loader and compute quality metrics."""
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
        ref = x_np[i]
        est = xhat_np[i]
        mse = float(np.mean((ref - est) ** 2))
        mae = float(np.mean(np.abs(ref - est)))
        rmse = float(np.sqrt(mse))
        snr_db = float(10.0 * np.log10((np.mean(ref ** 2) + 1e-8) / (np.mean((ref - est) ** 2) + 1e-8)))

        orig_std = float(np.std(ref))
        recon_std = float(np.std(est))
        orig_max = float(np.max(np.abs(ref)))
        recon_max = float(np.max(np.abs(est)))

        metrics.append({
            "sample": i,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "snr_db": snr_db,
            "si_snr_db": _si_snr_db(ref, est),
            "lsd_db": _lsd_db(ref, est),
            "envelope_corr": _envelope_corr(ref, est),
            "orig_std": orig_std,
            "recon_std": recon_std,
            "std_ratio": recon_std / (orig_std + 1e-8),
            "orig_max": orig_max,
            "recon_max": recon_max,
            "peak_ratio": recon_max / (orig_max + 1e-8),
        })

    summary = {}
    for key in ["mse", "mae", "rmse", "snr_db", "si_snr_db", "lsd_db", "envelope_corr", "std_ratio", "peak_ratio"]:
        vals = np.array([m[key] for m in metrics], dtype=float)
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    result = {
        "x_np": x_np,
        "xhat_np": xhat_np,
        "z": z.cpu().numpy(),
        "per_sample": metrics,
        "summary": summary,
    }

    if mu is not None:
        result["mu_mean"] = mu.mean().item()
        result["mu_std"] = mu.std().item()
        result["logvar_mean"] = logvar.mean().item()

    return result


def save_reconstruction_master_table(result: dict, output_dir: str, prefix: str = "reconstruction_master") -> dict:
    """Save thesis-friendly reconstruction table to CSV/Markdown/JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = ["mse", "mae", "rmse", "snr_db", "si_snr_db", "lsd_db", "envelope_corr", "std_ratio", "peak_ratio"]
    csv_path = out / f"{prefix}.csv"
    md_path = out / f"{prefix}.md"
    json_path = out / f"{prefix}.json"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("metric,mean,std\n")
        for k in keys:
            f.write(f"{k},{result['summary'][k]['mean']:.6f},{result['summary'][k]['std']:.6f}\n")

    lines = [
        "# Reconstruction Master Table",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
    ]
    for k in keys:
        s = result["summary"][k]
        lines.append(f"| {k} | {s['mean']:.6f} | {s['std']:.6f} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(result["summary"], f, indent=2)

    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "json": str(json_path),
    }


def print_metrics(result: dict):
    """Pretty-print reconstruction quality metrics."""
    print("Reconstruction Quality (Per Sample):")
    print("-" * 90)
    for m in result["per_sample"]:
        print(
            f"  Sample {m['sample']:2d}: "
            f"MSE={m['mse']:.5f} MAE={m['mae']:.5f} SI-SNR={m['si_snr_db']:.2f}dB "
            f"LSD={m['lsd_db']:.2f}dB EnvCorr={m['envelope_corr']:.3f} "
            f"StdRatio={m['std_ratio']:.2%}"
        )
    print("-" * 90)

    print("Summary (mean ± std):")
    for k, s in result["summary"].items():
        print(f"  {k:<13s}: {s['mean']:.6f} ± {s['std']:.6f}")

    if "mu_mean" in result:
        print(
            f"  Latent: mu mean={result['mu_mean']:.4f}  "
            f"mu std={result['mu_std']:.4f}  "
            f"logvar mean={result['logvar_mean']:.4f}"
        )
