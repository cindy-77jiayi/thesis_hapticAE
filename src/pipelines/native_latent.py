"""Helpers for analyzing and sweeping native latent axes without PCA."""

from __future__ import annotations

from pathlib import Path
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


def plot_sweep(
    sweep_result: dict,
    sr: int = 8000,
    save_path: str | None = None,
    overlay: bool = False,
) -> None:
    """Visualize a single native latent sweep."""
    import matplotlib.pyplot as plt

    values = sweep_result["values"]
    signals = sweep_result["signals"]
    axis_label = sweep_result.get("axis_label", f"z{sweep_result['axis'] + 1}")
    n = len(values)

    if overlay:
        base_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
        t = np.arange(len(signals[0])) / sr

        fig, ax = plt.subplots(figsize=(14, 4.5))
        for i, (val, sig) in enumerate(zip(values, signals)):
            color = base_colors[i % len(base_colors)]
            ax.plot(t, sig, linewidth=1.1, color=color, label=f"{val:+.2f}")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Single-axis sweep overlay: {axis_label} from {values[0]:.1f} to {values[-1]:.1f}")
        ax.set_ylim(-3.5, 3.5)
        ax.legend(title="Sweep value", ncol=min(3, n), fontsize=8, title_fontsize=9)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        try:
            import plotly.graph_objects as go

            interactive_fig = go.Figure()
            for i, (val, sig) in enumerate(zip(values, signals)):
                r, g, b = base_colors[i % len(base_colors)]
                interactive_fig.add_trace(
                    go.Scatter(
                        x=t,
                        y=sig,
                        mode="lines",
                        name=f"{val:+.2f}",
                        line={"color": f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})", "width": 2},
                    )
                )

            interactive_fig.update_layout(
                title=f"Interactive overlay: {axis_label}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                template="plotly_white",
                legend={
                    "title": {"text": "Sweep value"},
                    "orientation": "v",
                    "x": 1.02,
                    "y": 1.0,
                    "xanchor": "left",
                    "yanchor": "top",
                },
                margin={"l": 60, "r": 180, "t": 60, "b": 50},
            )
            interactive_fig.update_yaxes(range=[-3.5, 3.5])
            interactive_fig.show()

            if save_path:
                html_path = Path(save_path).with_suffix(".html")
                interactive_fig.write_html(str(html_path), include_plotlyjs="cdn")
        except Exception:
            pass

        return

    fig, axes = plt.subplots(n, 1, figsize=(14, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (val, sig) in enumerate(zip(values, signals)):
        t = np.arange(len(sig)) / sr
        axes[i].plot(t, sig, linewidth=0.5)
        axes[i].set_ylabel(f"{axis_label}={val:+.1f}", fontsize=9)
        axes[i].set_ylim(-3.5, 3.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Single-axis sweep: {axis_label} from {values[0]:.1f} to {values[-1]:.1f}", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def play_sweep(sweep_result: dict, sr: int = 8000) -> None:
    """Play sweep audio in Jupyter or Colab."""
    from IPython.display import Audio, display

    values = sweep_result["values"]
    signals = sweep_result["signals"]
    axis_label = sweep_result.get("axis_label", f"z{sweep_result['axis'] + 1}")

    for val, sig in zip(values, signals):
        sig_norm = sig / (np.max(np.abs(sig)) + 1e-8)
        sig_norm = np.clip(sig_norm, -1.0, 1.0)
        print(f"{axis_label} = {val:+.2f}  |  max={np.max(np.abs(sig)):.4f}")
        display(Audio(sig_norm, rate=sr))
