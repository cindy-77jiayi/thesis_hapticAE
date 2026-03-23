"""Quantitative evaluation of reconstruction quality."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval.signal_metrics import METRIC_GROUPS, compute_all_metrics


def _spectral_log_l1(x_hat: np.ndarray, x: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute difference in log-magnitude spectra."""
    X = np.abs(np.fft.rfft(x))
    Xh = np.abs(np.fft.rfft(x_hat))
    return float(np.mean(np.abs(np.log(X + eps) - np.log(Xh + eps))))


def _representative_metric_names() -> list[str]:
    """Ordered unique list of representative metrics across all groups."""
    names: list[str] = []
    for g in METRIC_GROUPS.values():
        for metric in g["representative"]:
            if metric not in names:
                names.append(metric)
    return names


# Floors prevent metric-relative errors from exploding when values are near zero.
_METRIC_REL_FLOOR = {
    "attack_time_s": 1e-3,
    "onset_density_ps": 0.1,
    "ioi_entropy_bits": 0.1,
}


def _stable_abs_relative_error(orig: float, recon: float, metric_name: str) -> float:
    """Stable relative error with metric-aware denominator floor."""
    floor = _METRIC_REL_FLOOR.get(metric_name, 1e-3)
    denom = max(abs(orig), floor)
    return abs(recon - orig) / denom


def _smape(orig: float, recon: float, metric_name: str) -> float:
    """Symmetric MAPE, bounded and robust for near-zero values."""
    floor = _METRIC_REL_FLOOR.get(metric_name, 1e-3)
    denom = abs(orig) + abs(recon) + floor
    return 2.0 * abs(recon - orig) / denom


def evaluate_reconstruction(
    model,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 10,
    is_vae: bool = True,
    sr: int = 8000,
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
    rep_names = _representative_metric_names()
    per_metric_rel_errors: dict[str, list[float]] = {k: [] for k in rep_names}
    per_metric_smapes: dict[str, list[float]] = {k: [] for k in rep_names}

    for i in range(len(x_np)):
        orig_std = np.std(x_np[i])
        recon_std = np.std(xhat_np[i])
        orig_max = np.max(np.abs(x_np[i]))
        recon_max = np.max(np.abs(xhat_np[i]))
        ratio = recon_std / (orig_std + 1e-8)
        mse = np.mean((xhat_np[i] - x_np[i]) ** 2)
        mae = np.mean(np.abs(xhat_np[i] - x_np[i]))
        spec_log_l1 = _spectral_log_l1(xhat_np[i], x_np[i])

        orig_sig = compute_all_metrics(x_np[i], sr=sr)
        recon_sig = compute_all_metrics(xhat_np[i], sr=sr)
        rep_delta = {}
        for name in rep_names:
            o = float(orig_sig[name])
            r = float(recon_sig[name])
            rel_err = _stable_abs_relative_error(o, r, name)
            smape = _smape(o, r, name)
            rep_delta[name] = {
                "orig": o,
                "recon": r,
                "abs_rel_err": rel_err,
                "smape": smape,
            }
            per_metric_rel_errors[name].append(rel_err)
            per_metric_smapes[name].append(smape)

        metrics.append({
            "sample": i,
            "orig_std": orig_std,
            "recon_std": recon_std,
            "std_ratio": ratio,
            "orig_max": orig_max,
            "recon_max": recon_max,
            "mse": float(mse),
            "mae": float(mae),
            "spectral_log_l1": float(spec_log_l1),
            "representative_metric_delta": rep_delta,
        })

    result = {
        "x_np": x_np,
        "xhat_np": xhat_np,
        "z": z.cpu().numpy(),
        "per_sample": metrics,
        "reconstruction_summary": {
            "mse_mean": float(np.mean([m["mse"] for m in metrics])),
            "mae_mean": float(np.mean([m["mae"] for m in metrics])),
            "spectral_log_l1_mean": float(np.mean([m["spectral_log_l1"] for m in metrics])),
            "representative_metric_abs_rel_err_mean": {
                name: float(np.mean(vals)) for name, vals in per_metric_rel_errors.items()
            },
            "representative_metric_smape_mean": {
                name: float(np.mean(vals)) for name, vals in per_metric_smapes.items()
            },
        },
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
            f"Orig Max={m['orig_max']:.4f}  Recon Max={m['recon_max']:.4f}  "
            f"MSE={m['mse']:.5f}  MAE={m['mae']:.5f}  SpecLogL1={m['spectral_log_l1']:.5f}"
        )
    print("-" * 70)
    if "mu_mean" in result:
        print(
            f"  Latent: mu mean={result['mu_mean']:.4f}  "
            f"mu std={result['mu_std']:.4f}  "
            f"logvar mean={result['logvar_mean']:.4f}"
        )

    s = result["reconstruction_summary"]
    print(
        f"  Summary: MSE={s['mse_mean']:.6f}  "
        f"MAE={s['mae_mean']:.6f}  SpecLogL1={s['spectral_log_l1_mean']:.6f}"
    )
    print("  Representative Metric Abs-Relative-Error (mean):")
    rep = s["representative_metric_abs_rel_err_mean"]
    for name in _representative_metric_names():
        print(f"    - {name}: {rep[name]:.2%}")
    print("  Representative Metric sMAPE (mean, bounded):")
    rep_smape = s["representative_metric_smape_mean"]
    for name in _representative_metric_names():
        print(f"    - {name}: {rep_smape[name]:.2%}")
