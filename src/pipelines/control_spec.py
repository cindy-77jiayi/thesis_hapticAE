"""Build the frozen control specification: ranges, metric profiles, and labels."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline

from src.pipelines.pca_control import sweep_axis


METRIC_LABELS = {
    "rms_energy": ("RMS energy", ""),
    "peak_amplitude": ("Peak amplitude", ""),
    "spectral_centroid_hz": ("Spectral centroid", "Hz"),
    "spectral_rolloff_hz": ("Spectral rolloff", "Hz"),
    "spectral_slope": ("Spectral slope", "dB/Hz"),
    "spectral_flatness": ("Spectral flatness", ""),
    "high_freq_ratio": ("HF energy ratio", ""),
    "low_high_band_ratio": ("Low/High band ratio", ""),
    "band_energy_0_150": ("Band 0-150 Hz", ""),
    "band_energy_150_400": ("Band 150-400 Hz", ""),
    "band_energy_400_800": ("Band 400-800 Hz", ""),
    "envelope_decay_slope_dBps": ("Envelope decay slope", "dB/s"),
    "late_early_energy_ratio": ("Late/Early energy ratio", ""),
    "attack_time_s": ("Attack time", "s"),
    "transient_energy_ratio": ("Transient energy ratio", ""),
    "effective_duration_s": ("Effective duration", "s"),
    "envelope_area": ("Envelope area", ""),
    "envelope_entropy_bits": ("Envelope entropy", "bits"),
    "onset_density_ps": ("Onset density", "/s"),
    "ioi_entropy_bits": ("IOI entropy", "bits"),
    "onset_interval_cv": ("Onset interval CV", ""),
    "modulation_peak_hz": ("Modulation peak", "Hz"),
    "zero_crossing_rate_ps": ("Zero-crossing rate", "/s"),
    "gap_ratio": ("Gap ratio", ""),
    "short_term_variance": ("Short-term variance", ""),
    "am_modulation_index": ("AM modulation index", ""),
    "crest_factor": ("Crest factor", ""),
}


# ---------------------------------------------------------------------------
# Optional PC labels
# ---------------------------------------------------------------------------

DEFAULT_PC_LABELS: dict[int, dict] = {}


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


def _promote_primary_metric(
    dominant: list[dict],
    metric_usage: dict[str, int],
    max_primary_reuse: int | None,
) -> list[dict]:
    """Reorder dominant metrics so the first one is less redundant across PCs.

    The first entry is used as the "primary" metric in control summaries.
    If max_primary_reuse is set, we prefer metrics whose current usage is below
    that cap; otherwise keep the original ranking.
    """
    if not dominant or max_primary_reuse is None:
        if dominant:
            metric_usage[dominant[0]["metric"]] = metric_usage.get(dominant[0]["metric"], 0) + 1
        return dominant

    chosen_idx = None
    for idx, item in enumerate(dominant):
        name = item["metric"]
        if metric_usage.get(name, 0) < max_primary_reuse:
            chosen_idx = idx
            break

    if chosen_idx is None:
        chosen_idx = 0

    if chosen_idx != 0:
        dominant = [dominant[chosen_idx], *dominant[:chosen_idx], *dominant[chosen_idx + 1:]]

    metric_usage[dominant[0]["metric"]] = metric_usage.get(dominant[0]["metric"], 0) + 1
    return dominant


def build_controls_spec(
    pipe: Pipeline,
    model,
    device: torch.device,
    Z_pca: np.ndarray,
    explained_variance_ratio: np.ndarray,
    T: int = 4000,
    sr: int = 8000,
    n_sweep_steps: int = 11,
    max_primary_reuse: int | None = 2,
) -> dict:
    """Build the complete control specification with ranges and metric profiles.

    Returns a JSON-serializable dict.
    """
    n_components = Z_pca.shape[1]
    ranges = compute_control_ranges(Z_pca)

    controls = []
    metric_usage: dict[str, int] = {}
    for i in range(n_components):
        r = ranges[i]
        sweep = sweep_axis(
            pipe, model, device,
            axis=i,
            sweep_range=(r["p5"], r["p95"]),
            n_steps=n_sweep_steps,
            T=T, sr=sr,
            with_metrics=True,
        )
        dominant = identify_dominant_metrics(sweep)
        dominant = _promote_primary_metric(
            dominant,
            metric_usage=metric_usage,
            max_primary_reuse=max_primary_reuse,
        )

        ctrl_entry = {
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
        }
        if i in DEFAULT_PC_LABELS:
            lbl = DEFAULT_PC_LABELS[i]
            ctrl_entry["composite_label"] = lbl["label"]
            ctrl_entry["composite_description"] = lbl["description"]
            ctrl_entry["adjectives"] = lbl.get("adjectives", [])
            ctrl_entry["key_metrics"] = lbl.get("key_metrics", {})
        controls.append(ctrl_entry)

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
    """Generate controls_table.md content with composite labels.

    Args:
        spec: The controls_spec dict (may contain composite_label fields).
        pc_labels: Optional override dict mapping axis index to
            {"label": str, "description": str, "adjectives": [str, ...]}.
            If None, uses composite labels embedded in the spec, falling back
            to DEFAULT_PC_LABELS, then to the top dominant metric.
    """
    n = spec["n_controls"]
    lines = [
        f"# Haptic Control Dimensions (PC1\u2013PC{n})",
        "",
        f"Total explained variance: **{spec['total_explained_variance_pct']:.1f}%**  ",
        f"Signal: {spec['signal_length']} samples @ {spec['sr']} Hz "
        f"({spec['signal_length']/spec['sr']:.2f}s)",
        "",
        "## Control Summary",
        "",
        "| Control | Var% | Range (P5\u2013P95) | Composite Label | Low \u2192 High |",
        "|---------|------|----------------|----------------|------------|",
    ]

    for ctrl in spec["controls"]:
        ax = ctrl["axis"]
        r = ctrl["range"]

        label, adj_lo, adj_hi = _resolve_label(ctrl, ax, pc_labels)

        lines.append(
            f"| {ctrl['name']} | {ctrl['explained_variance_pct']:.1f} | "
            f"[{r['low']:+.2f}, {r['high']:+.2f}] | "
            f"{label} | {adj_lo} \u2192 {adj_hi} |"
        )

    lines.append("")
    lines.append("## Detailed Metric Profiles")
    lines.append("")

    for ctrl in spec["controls"]:
        ax = ctrl["axis"]
        label, adj_lo, adj_hi = _resolve_label(ctrl, ax, pc_labels)
        desc = _resolve_description(ctrl, ax, pc_labels)

        lines.append(
            f"### {ctrl['name']} \u2014 {label} "
            f"({ctrl['explained_variance_pct']:.1f}% variance)"
        )
        lines.append("")
        lines.append(f"**Description:** {desc}  ")
        lines.append(f"**Subjective:** low \u2192 _{adj_lo}_, high \u2192 _{adj_hi}_")
        lines.append("")
        lines.append(
            f"- Range (P5\u2013P95): [{ctrl['range']['low']:+.2f}, "
            f"{ctrl['range']['high']:+.2f}]"
        )
        lines.append(
            f"- Data mean: {ctrl['statistics']['mean']:.4f}, "
            f"std: {ctrl['statistics']['std']:.4f}"
        )
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


def _resolve_label(
    ctrl: dict, ax: int, pc_labels: dict[int, dict] | None
) -> tuple[str, str, str]:
    """Return (label, adj_low, adj_high) from the best available source."""
    if pc_labels and ax in pc_labels:
        lbl = pc_labels[ax]
    elif "composite_label" in ctrl:
        lbl = {
            "label": ctrl["composite_label"],
            "adjectives": ctrl.get("adjectives", []),
        }
    elif ax in DEFAULT_PC_LABELS:
        lbl = DEFAULT_PC_LABELS[ax]
    else:
        dom = ctrl["dominant_metrics"][0] if ctrl["dominant_metrics"] else {}
        metric_name = METRIC_LABELS.get(
            dom.get("metric", ""), (dom.get("metric", "?"), "")
        )[0]
        return metric_name, "low", "high"

    adjs = lbl.get("adjectives", ["low", "high"])
    return lbl["label"], adjs[0] if len(adjs) > 0 else "low", adjs[1] if len(adjs) > 1 else "high"


def _resolve_description(
    ctrl: dict, ax: int, pc_labels: dict[int, dict] | None
) -> str:
    """Return the best available description string."""
    if pc_labels and ax in pc_labels:
        return pc_labels[ax].get("description", "")
    if "composite_description" in ctrl:
        return ctrl["composite_description"]
    if ax in DEFAULT_PC_LABELS:
        return DEFAULT_PC_LABELS[ax].get("description", "")
    dom = ctrl["dominant_metrics"][0] if ctrl["dominant_metrics"] else {}
    metric_name = METRIC_LABELS.get(
        dom.get("metric", ""), (dom.get("metric", "?"), "")
    )[0]
    direction = dom.get("direction", "?")
    return f"Primary effect: {metric_name} {direction}"


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
