"""Helpers for analyzing and sweeping native latent axes without PCA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.eval.signal_metrics import compute_all_metrics

if TYPE_CHECKING:
    import torch


def summarize_latent_dimensions(
    Z: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> list[dict[str, float]]:
    """Return per-axis descriptive statistics for native latent vectors."""
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected Z to have shape (N, D), got {Z.shape}")

    quantile_values = np.quantile(Z, quantiles, axis=0)
    rows: list[dict[str, float]] = []
    for axis in range(Z.shape[1]):
        row: dict[str, float] = {
            "axis": axis,
            "axis_label": f"z{axis + 1}",
            "mean": float(np.mean(Z[:, axis])),
            "std": float(np.std(Z[:, axis])),
            "min": float(np.min(Z[:, axis])),
            "max": float(np.max(Z[:, axis])),
        }
        for q, value in zip(quantiles, quantile_values[:, axis], strict=True):
            key = f"q{int(round(q * 100)):02d}"
            row[key] = float(value)
        rows.append(row)
    return rows


def compute_latent_ranges(
    Z: np.ndarray,
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> list[dict[str, float]]:
    """Return sweep ranges for each native latent axis from empirical quantiles."""
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"Expected Z to have shape (N, D), got {Z.shape}")
    if not 0.0 <= low_q < high_q <= 1.0:
        raise ValueError("Expected 0 <= low_q < high_q <= 1")

    lows = np.quantile(Z, low_q, axis=0)
    highs = np.quantile(Z, high_q, axis=0)
    return [
        {
            "axis": axis,
            "axis_label": f"z{axis + 1}",
            "low": float(lows[axis]),
            "high": float(highs[axis]),
            "mean": float(np.mean(Z[:, axis])),
            "std": float(np.std(Z[:, axis])),
        }
        for axis in range(Z.shape[1])
    ]


def decode_latent_vector(
    model,
    device: "torch.device",
    z: np.ndarray,
    T: int,
) -> np.ndarray:
    """Decode a single latent vector into a waveform."""
    import torch

    z = np.asarray(z, dtype=np.float32).reshape(1, -1)
    z_t = torch.from_numpy(z).to(device)
    model.eval()
    with torch.no_grad():
        x_hat = model.decode(z_t, target_len=T)
    return x_hat.squeeze().detach().cpu().numpy()


def sweep_latent_axis(
    model,
    device: "torch.device",
    axis: int,
    sweep_range: tuple[float, float] = (-2.0, 2.0),
    n_steps: int = 9,
    T: int = 4000,
    sr: int = 8000,
    reference: np.ndarray | None = None,
    latent_dim: int | None = None,
    with_metrics: bool = False,
) -> dict:
    """Sweep a native latent axis and decode the resulting signals."""
    if reference is None and latent_dim is None:
        raise ValueError("Provide either a reference vector or latent_dim")

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32).reshape(-1)
        dim = ref.shape[0]
    else:
        dim = int(latent_dim)
        ref = np.zeros(dim, dtype=np.float32)

    if axis < 0 or axis >= dim:
        raise IndexError(f"Axis {axis} is out of bounds for latent_dim={dim}")

    values = np.linspace(sweep_range[0], sweep_range[1], n_steps, dtype=np.float32)
    signals: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    metrics_list: list[dict] = []

    for value in values:
        z = ref.copy()
        z[axis] = float(value)
        signal = decode_latent_vector(model, device, z, T=T)
        latents.append(z)
        signals.append(signal)
        if with_metrics:
            metrics_list.append(compute_all_metrics(signal, sr=sr))

    result = {
        "axis": int(axis),
        "axis_label": f"z{axis + 1}",
        "values": values if not with_metrics else values.tolist(),
        "signals": np.stack(signals),
        "latents": np.stack(latents),
    }
    if with_metrics:
        result["metrics"] = metrics_list
    return result
