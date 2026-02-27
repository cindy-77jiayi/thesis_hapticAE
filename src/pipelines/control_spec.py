"""Build the frozen control specification: ranges, metric profiles, and labels."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline

from src.pipelines.pca_control import control_to_latent
from src.eval.signal_metrics import compute_all_metrics


METRIC_LABELS = {
    "rms_energy": ("RMS energy", ""),
    "peak_amplitude": ("Peak amplitude", ""),
    "spectral_centroid_hz": ("Spectral centroid", "Hz"),
    "high_freq_ratio": ("HF energy ratio", ""),
    "low_high_band_ratio": ("Low/High band ratio", ""),
    "envelope_decay_slope_dBps": ("Envelope decay slope", "dB/s"),
    "late_early_energy_ratio": ("Late/Early energy ratio", ""),
    "attack_time_s": ("Attack time", "s"),
    "onset_density_ps": ("Onset density", "/s"),
    "ioi_entropy_bits": ("IOI entropy", "bits"),
    "zero_crossing_rate_ps": ("Zero-crossing rate", "/s"),
    "gap_ratio": ("Gap ratio", ""),
    "short_term_variance": ("Short-term variance", ""),
    "am_modulation_index": ("AM modulation index", ""),
    "crest_factor": ("Crest factor", ""),
}


def compute_control_ranges(
    Z_pca: np.ndarray,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> list[dict]:
    """Compute per-axis statistics and safe ranges from training data.

    Returns a list of dicts, one per PC, with keys: axis, p5, p95, mean, std, min, max.
    """
    n_components = Z_pca.shape[1]
    ranges = []
    for i in range(n_components):
        col = Z_pca[:, i]
        lo, hi = np.percentile(col, percentiles[0]), np.percentile(col, percentiles[1])
        ranges.append({
            "axis": i,
            "pc": f"PC{i + 1}",
            "p5": round(float(lo), 4),
            "p95": round(float(hi), 4),
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "min": round(float(col.min()), 4),
            "max": round(float(col.max()), 4),
        })
    return ranges


def sweep_with_metrics(
    pipe: Pipeline,
    model,
    device: torch.device,
    axis: int,
    sweep_range: tuple[float, float],
    n_steps: int = 11,
    T: int = 4000,
    sr: int = 8000,
) -> dict:
    """Sweep one PC axis and compute signal metrics at each step.

    Returns dict with 'values', 'signals', 'metrics' (list of metric dicts).
    """
    n_components = pipe.named_steps["pca"].n_components
    values = np.linspace(sweep_range[0], sweep_range[1], n_steps)

    signals = []
    metrics_list = []

    model.eval()
    with torch.no_grad():
        for val in values:
            c = np.zeros(n_components, dtype=np.float32)
            c[axis] = val

            z_np = control_to_latent(pipe, c)
            z_t = torch.from_numpy(z_np).float().unsqueeze(0).to(device)
            x_hat = model.decode(z_t, target_len=T)
            sig = x_hat.squeeze().cpu().numpy()

            signals.append(sig)
            metrics_list.append(compute_all_metrics(sig, sr=sr))

    return {
        "axis": axis,
        "values": values.tolist(),
        "signals": np.stack(signals),
        "metrics": metrics_list,
    }


def _trend_direction(values: list[float]) -> str:
    """Determine if a metric increases, decreases, or stays flat over the sweep."""
    arr = np.array(values)
    if len(arr) < 3:
        return "flat"
    coeffs = np.polyfit(np.arange(len(arr)), arr, deg=1)
    slope = coeffs[0]
    span = arr.max() - arr.min()
    mean_abs = np.mean(np.abs(arr)) + 1e-10
    relative_change = span / mean_abs
    if relative_change < 0.15:
        return "flat"
    return "increases" if slope > 0 else "decreases"


def identify_dominant_metrics(sweep_result: dict, top_k: int = 3) -> list[dict]:
    """Find which metrics change most strongly across the sweep.

    Returns top_k metrics sorted by relative change, with direction.
    """
    metrics_list = sweep_result["metrics"]
    metric_names = list(metrics_list[0].keys())

    changes = []
    for name in metric_names:
        vals = [m[name] for m in metrics_list]
        arr = np.array(vals)
        span = arr.max() - arr.min()
        mean_abs = np.mean(np.abs(arr)) + 1e-10
        relative_change = span / mean_abs
        direction = _trend_direction(vals)

        changes.append({
            "metric": name,
            "direction": direction,
            "relative_change": round(float(relative_change), 4),
            "range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
        })

    changes.sort(key=lambda x: x["relative_change"], reverse=True)
    return changes[:top_k]


def build_controls_spec(
    pipe: Pipeline,
    model,
    device: torch.device,
    Z_pca: np.ndarray,
    explained_variance_ratio: np.ndarray,
    T: int = 4000,
    sr: int = 8000,
    n_sweep_steps: int = 11,
) -> dict:
    """Build the complete control specification with ranges and metric profiles.

    Returns a JSON-serializable dict.
    """
    n_components = Z_pca.shape[1]
    ranges = compute_control_ranges(Z_pca)

    controls = []
    for i in range(n_components):
        r = ranges[i]
        sweep = sweep_with_metrics(
            pipe, model, device,
            axis=i,
            sweep_range=(r["p5"], r["p95"]),
            n_steps=n_sweep_steps,
            T=T, sr=sr,
        )
        dominant = identify_dominant_metrics(sweep)

        controls.append({
            "axis": i,
            "name": f"PC{i + 1}",
            "explained_variance_pct": round(float(explained_variance_ratio[i]) * 100, 2),
            "range": {"low": r["p5"], "high": r["p95"]},
            "default": 0.0,
            "statistics": {
                "mean": r["mean"],
                "std": r["std"],
                "data_min": r["min"],
                "data_max": r["max"],
            },
            "dominant_metrics": dominant,
        })

    total_var = sum(c["explained_variance_pct"] for c in controls)
    spec = {
        "n_controls": n_components,
        "total_explained_variance_pct": round(total_var, 2),
        "sr": sr,
        "signal_length": T,
        "controls": controls,
    }
    return spec


def save_controls_spec(spec: dict, path: str):
    """Save controls_spec.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Saved: {p}")


def build_controls_table_md(spec: dict, pc_labels: dict[int, dict] | None = None) -> str:
    """Generate controls_table.md content.

    Args:
        spec: The controls_spec dict.
        pc_labels: Optional dict mapping axis index to
            {"label": str, "description": str, "adjectives": [str, ...]}.
            If None, labels are auto-generated from dominant metrics.
    """
    lines = [
        "# Haptic Control Dimensions (PC1–PC8)",
        "",
        f"Total explained variance: **{spec['total_explained_variance_pct']:.1f}%**  ",
        f"Signal: {spec['signal_length']} samples @ {spec['sr']} Hz ({spec['signal_length']/spec['sr']:.2f}s)",
        "",
        "## Control Summary",
        "",
        "| Control | Var% | Range (P5–P95) | Primary Effect | Direction |",
        "|---------|------|----------------|---------------|-----------|",
    ]

    for ctrl in spec["controls"]:
        ax = ctrl["axis"]
        r = ctrl["range"]
        dom = ctrl["dominant_metrics"][0] if ctrl["dominant_metrics"] else {}
        metric_name = METRIC_LABELS.get(dom.get("metric", ""), (dom.get("metric", "?"), ""))[0]
        direction = dom.get("direction", "?")

        if pc_labels and ax in pc_labels:
            label = pc_labels[ax]["label"]
        else:
            label = metric_name

        lines.append(
            f"| {ctrl['name']} | {ctrl['explained_variance_pct']:.1f} | "
            f"[{r['low']:+.2f}, {r['high']:+.2f}] | "
            f"{label} | {direction} |"
        )

    lines.append("")
    lines.append("## Detailed Metric Profiles")
    lines.append("")

    for ctrl in spec["controls"]:
        ax = ctrl["axis"]
        lines.append(f"### {ctrl['name']} — {ctrl['explained_variance_pct']:.1f}% variance")

        if pc_labels and ax in pc_labels:
            lbl = pc_labels[ax]
            lines.append(f"**Label:** {lbl['label']}  ")
            lines.append(f"**Description:** {lbl['description']}  ")
            if lbl.get("adjectives"):
                lines.append(f"**Subjective:** low → {lbl['adjectives'][0]}, high → {lbl['adjectives'][1]}")
        lines.append("")

        lines.append(f"- Range (P5–P95): [{ctrl['range']['low']:+.2f}, {ctrl['range']['high']:+.2f}]")
        lines.append(f"- Data mean: {ctrl['statistics']['mean']:.4f}, std: {ctrl['statistics']['std']:.4f}")
        lines.append("")

        lines.append("| Metric | Direction | Relative Change | Range |")
        lines.append("|--------|-----------|-----------------|-------|")
        for dm in ctrl["dominant_metrics"]:
            mname, unit = METRIC_LABELS.get(dm["metric"], (dm["metric"], ""))
            unit_str = f" {unit}" if unit else ""
            lines.append(
                f"| {mname} | {dm['direction']} | "
                f"{dm['relative_change']:.2f}x | "
                f"[{dm['range'][0]:.3f}, {dm['range'][1]:.3f}]{unit_str} |"
            )
        lines.append("")

    return "\n".join(lines)


def plot_sweep_gallery(
    sweep_results: list[dict],
    sr: int = 8000,
    save_dir: str | None = None,
):
    """Plot waveform + spectrogram for every PC sweep. Saves individual PNGs."""
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for sweep in sweep_results:
        axis = sweep["axis"]
        values = sweep["values"]
        signals = sweep["signals"]
        n = len(values)

        fig, axes = plt.subplots(n, 2, figsize=(16, 1.6 * n), gridspec_kw={"width_ratios": [2, 1]})
        if n == 1:
            axes = [axes]

        for i, (val, sig) in enumerate(zip(values, signals)):
            t = np.arange(len(sig)) / sr

            axes[i][0].plot(t, sig, linewidth=0.4, color="steelblue")
            axes[i][0].set_ylabel(f"c={val:+.2f}", fontsize=8)
            axes[i][0].set_ylim(-3.5, 3.5)
            axes[i][0].tick_params(labelsize=7)

            axes[i][1].magnitude_spectrum(sig, Fs=sr, scale="dB", color="coral", linewidth=0.6)
            axes[i][1].set_xlim(0, sr / 2)
            axes[i][1].set_ylabel("dB", fontsize=7)
            axes[i][1].tick_params(labelsize=7)
            axes[i][1].set_title("")

        axes[-1][0].set_xlabel("Time (s)")
        axes[-1][1].set_xlabel("Frequency (Hz)")
        fig.suptitle(f"PC{axis + 1} sweep", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_dir:
            fname = save_dir / f"sweep_PC{axis + 1}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fname}")

        plt.show()


def plot_metric_trends(
    sweep_results: list[dict],
    save_dir: str | None = None,
):
    """Plot how each metric changes across each PC sweep."""
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    metric_names = list(sweep_results[0]["metrics"][0].keys())
    n_pcs = len(sweep_results)
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 2.2 * n_metrics), sharex=False)
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_pcs))

    for mi, mname in enumerate(metric_names):
        ax = axes[mi]
        for si, sweep in enumerate(sweep_results):
            vals = sweep["values"]
            metric_vals = [m[mname] for m in sweep["metrics"]]
            ax.plot(vals, metric_vals, "o-", markersize=3, linewidth=1.2,
                    color=colors[si], label=f"PC{sweep['axis'] + 1}")

        label, unit = METRIC_LABELS.get(mname, (mname, ""))
        unit_str = f" ({unit})" if unit else ""
        ax.set_ylabel(f"{label}{unit_str}", fontsize=9)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Control value")
    fig.suptitle("Metric trends across PC sweeps", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        fname = save_dir / "metric_trends.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fname}")

    plt.show()
