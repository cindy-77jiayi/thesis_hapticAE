"""Statistical validation of PC control dimensions.

Produces three categories of evidence:
  1. Monotonicity: Spearman ρ for each (PC, metric) pair
  2. Orthogonality: cross-influence matrix showing each PC primarily
     affects its bound metrics, not others'
  3. Effect size: Cohen's d or relative change for PC1-4 vs PC5-8
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Monotonicity: Spearman correlation matrix
# ---------------------------------------------------------------------------

def compute_monotonicity_matrix(sweep_results: list[dict]) -> dict:
    """Compute Spearman ρ and p-value for each (PC, metric) pair.

    Args:
        sweep_results: list of dicts from sweep_with_metrics(), one per PC.
            Each must have 'values' (list[float]) and 'metrics' (list[dict]).

    Returns:
        dict with:
          'metric_names': list[str]
          'pc_names': list[str]
          'rho': np.ndarray (n_pcs, n_metrics) — Spearman ρ
          'pvalue': np.ndarray (n_pcs, n_metrics) — p-values
    """
    metric_names = list(sweep_results[0]["metrics"][0].keys())
    n_pcs = len(sweep_results)
    n_metrics = len(metric_names)

    rho = np.zeros((n_pcs, n_metrics))
    pval = np.zeros((n_pcs, n_metrics))

    for pi, sweep in enumerate(sweep_results):
        values = np.array(sweep["values"])
        for mi, mname in enumerate(metric_names):
            metric_vals = np.array([m[mname] for m in sweep["metrics"]])
            if np.std(metric_vals) < 1e-12:
                rho[pi, mi] = 0.0
                pval[pi, mi] = 1.0
            else:
                r, p = spearmanr(values, metric_vals)
                rho[pi, mi] = r
                pval[pi, mi] = p

    return {
        "metric_names": metric_names,
        "pc_names": [f"PC{s['axis'] + 1}" for s in sweep_results],
        "rho": rho,
        "pvalue": pval,
    }


def print_monotonicity_report(mono: dict, sig_threshold: float = 0.05):
    """Print a readable monotonicity report."""
    metric_names = mono["metric_names"]
    pc_names = mono["pc_names"]
    rho = mono["rho"]
    pval = mono["pvalue"]

    print("\n" + "=" * 70)
    print("MONOTONICITY: Spearman ρ (PC × Metric)")
    print("=" * 70)

    header = f"{'':>12s}"
    for mn in metric_names:
        short = mn[:16]
        header += f" {short:>16s}"
    print(header)
    print("-" * len(header))

    for pi, pc in enumerate(pc_names):
        row = f"{pc:>12s}"
        for mi in range(len(metric_names)):
            r = rho[pi, mi]
            p = pval[pi, mi]
            star = "*" if p < sig_threshold else " "
            row += f" {r:>+7.3f}{star:>1s}       "
        print(row)

    print(f"\n* = p < {sig_threshold}")

    print("\nStrong monotonic pairs (|ρ| > 0.8, p < 0.05):")
    for pi, pc in enumerate(pc_names):
        strong = []
        for mi, mn in enumerate(metric_names):
            if abs(rho[pi, mi]) > 0.8 and pval[pi, mi] < sig_threshold:
                direction = "↑" if rho[pi, mi] > 0 else "↓"
                strong.append(f"{mn}({direction}, ρ={rho[pi, mi]:+.3f})")
        if strong:
            print(f"  {pc}: {', '.join(strong)}")


# ---------------------------------------------------------------------------
# 2. Orthogonality: cross-influence analysis
# ---------------------------------------------------------------------------

def compute_cross_influence(mono: dict, bindings: dict[str, list[str]]) -> dict:
    """Measure how much each PC affects its own bound metrics vs others'.

    Args:
        mono: output of compute_monotonicity_matrix()
        bindings: dict mapping PC name to list of bound metric names,
            e.g. {"PC1": ["rms_energy", "peak_amplitude"], ...}

    Returns:
        dict with per-PC analysis:
          'on_target_rho': mean |ρ| for bound metrics
          'off_target_rho': mean |ρ| for unbound metrics
          'selectivity': on_target / off_target ratio
    """
    rho = mono["rho"]
    metric_names = mono["metric_names"]
    pc_names = mono["pc_names"]

    results = {}
    for pi, pc in enumerate(pc_names):
        bound = bindings.get(pc, [])
        bound_idx = [mi for mi, mn in enumerate(metric_names) if mn in bound]
        unbound_idx = [mi for mi, mn in enumerate(metric_names) if mn not in bound]

        on_target = np.mean(np.abs(rho[pi, bound_idx])) if bound_idx else 0.0
        off_target = np.mean(np.abs(rho[pi, unbound_idx])) if unbound_idx else 0.0

        results[pc] = {
            "bound_metrics": bound,
            "on_target_mean_abs_rho": round(float(on_target), 4),
            "off_target_mean_abs_rho": round(float(off_target), 4),
            "selectivity": round(float(on_target / (off_target + 1e-8)), 2),
        }

    return results


def print_cross_influence_report(cross: dict):
    """Print the cross-influence / selectivity report."""
    print("\n" + "=" * 70)
    print("ORTHOGONALITY: Cross-influence analysis")
    print("=" * 70)
    print(f"{'PC':>6s}  {'On-target |ρ|':>14s}  {'Off-target |ρ|':>15s}  {'Selectivity':>12s}  Bound metrics")
    print("-" * 90)
    for pc, info in cross.items():
        bound_str = ", ".join(info["bound_metrics"][:3])
        print(f"{pc:>6s}  {info['on_target_mean_abs_rho']:>14.3f}  "
              f"{info['off_target_mean_abs_rho']:>15.3f}  "
              f"{info['selectivity']:>12.1f}x  {bound_str}")

    avg_sel = np.mean([v["selectivity"] for v in cross.values()])
    print(f"\nMean selectivity: {avg_sel:.1f}x (>2x = good orthogonality)")


# ---------------------------------------------------------------------------
# 3. Effect size: PC1-4 vs PC5-8
# ---------------------------------------------------------------------------

def compute_effect_sizes(sweep_results: list[dict]) -> dict:
    """Compute per-PC effect sizes (absolute metric range) for each metric.

    Returns dict[pc_name] -> dict[metric_name] -> {'range': float, 'relative_change': float}
    """
    metric_names = list(sweep_results[0]["metrics"][0].keys())
    results = {}

    for sweep in sweep_results:
        pc = f"PC{sweep['axis'] + 1}"
        effects = {}
        for mn in metric_names:
            vals = np.array([m[mn] for m in sweep["metrics"]])
            span = float(vals.max() - vals.min())
            mean_abs = float(np.mean(np.abs(vals))) + 1e-10
            effects[mn] = {
                "range": round(span, 6),
                "relative_change": round(span / mean_abs, 4),
            }
        results[pc] = effects

    return results


def print_effect_size_report(effects: dict, primary_pcs: list[str] = None, secondary_pcs: list[str] = None):
    """Compare effect sizes between primary (PC1-4) and secondary (PC5-8) PCs."""
    if primary_pcs is None:
        primary_pcs = ["PC1", "PC2", "PC3", "PC4"]
    if secondary_pcs is None:
        secondary_pcs = ["PC5", "PC6", "PC7", "PC8"]

    metric_names = list(list(effects.values())[0].keys())

    print("\n" + "=" * 70)
    print("EFFECT SIZE: Primary (PC1-4) vs Secondary (PC5-8)")
    print("=" * 70)

    print(f"\n{'Metric':<30s}  {'Primary avg':>12s}  {'Secondary avg':>14s}  {'Ratio':>8s}")
    print("-" * 70)

    for mn in metric_names:
        primary_vals = [effects[pc][mn]["relative_change"] for pc in primary_pcs if pc in effects]
        secondary_vals = [effects[pc][mn]["relative_change"] for pc in secondary_pcs if pc in effects]
        p_avg = np.mean(primary_vals) if primary_vals else 0
        s_avg = np.mean(secondary_vals) if secondary_vals else 0
        ratio = p_avg / (s_avg + 1e-8)
        short_name = mn[:28]
        print(f"{short_name:<30s}  {p_avg:>12.3f}  {s_avg:>14.3f}  {ratio:>7.1f}x")


# ---------------------------------------------------------------------------
# 4. Plot: correlation heatmap
# ---------------------------------------------------------------------------

def plot_monotonicity_heatmap(mono: dict, save_path: str | None = None):
    """Plot Spearman ρ as a heatmap with significance markers."""
    import matplotlib.pyplot as plt

    rho = mono["rho"]
    pval = mono["pvalue"]
    metric_names = mono["metric_names"]
    pc_names = mono["pc_names"]

    short_names = [n.replace("_dBps", "").replace("_ps", "").replace("_hz", "")
                   .replace("_bits", "").replace("_s", "")[:18] for n in metric_names]

    fig, ax = plt.subplots(figsize=(max(14, len(metric_names) * 0.9), max(6, len(pc_names) * 0.7)))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for pi in range(len(pc_names)):
        for mi in range(len(metric_names)):
            r = rho[pi, mi]
            p = pval[pi, mi]
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            color = "white" if abs(r) > 0.5 else "black"
            ax.text(mi, pi, f"{r:+.2f}{star}", ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pc_names)))
    ax.set_yticklabels(pc_names, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman ρ", fontsize=10)

    ax.set_title("Monotonicity: Spearman ρ (PC × Metric)\n* p<0.05  ** p<0.01", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    plt.show()


def plot_selectivity_bar(cross: dict, save_path: str | None = None):
    """Bar chart comparing on-target vs off-target |ρ| for each PC."""
    import matplotlib.pyplot as plt

    pcs = list(cross.keys())
    on = [cross[pc]["on_target_mean_abs_rho"] for pc in pcs]
    off = [cross[pc]["off_target_mean_abs_rho"] for pc in pcs]

    x = np.arange(len(pcs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, on, w, label="On-target (bound metrics)", color="steelblue")
    ax.bar(x + w / 2, off, w, label="Off-target (other metrics)", color="salmon")

    ax.set_xticks(x)
    ax.set_xticklabels(pcs)
    ax.set_ylabel("Mean |Spearman ρ|")
    ax.set_title("Selectivity: Each PC should primarily affect its bound metrics")
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, pc in enumerate(pcs):
        sel = cross[pc]["selectivity"]
        ax.text(i, max(on[i], off[i]) + 0.03, f"{sel:.1f}×", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 5. Cross-seed stability comparison
# ---------------------------------------------------------------------------

def compare_cross_seed(seed_results: list[dict]) -> dict:
    """Compare PCA results across different training seeds.

    Args:
        seed_results: list of dicts, each with:
            'seed': int
            'explained_variance_ratio': np.ndarray (n_components,)
            'mono': output of compute_monotonicity_matrix()

    Returns:
        dict with stability metrics.
    """
    n_seeds = len(seed_results)
    n_pcs = len(seed_results[0]["explained_variance_ratio"])

    evrs = np.array([sr["explained_variance_ratio"] for sr in seed_results])
    evr_mean = evrs.mean(axis=0)
    evr_std = evrs.std(axis=0)

    metric_names = seed_results[0]["mono"]["metric_names"]
    n_metrics = len(metric_names)

    sign_agreement = np.zeros((n_pcs, n_metrics))
    for mi in range(n_metrics):
        for pi in range(n_pcs):
            signs = [np.sign(sr["mono"]["rho"][pi, mi]) for sr in seed_results]
            if all(s == signs[0] for s in signs) and signs[0] != 0:
                sign_agreement[pi, mi] = 1.0
            elif sum(s == signs[0] for s in signs) >= n_seeds - 1:
                sign_agreement[pi, mi] = 0.5

    return {
        "n_seeds": n_seeds,
        "seeds": [sr["seed"] for sr in seed_results],
        "evr_mean": evr_mean,
        "evr_std": evr_std,
        "sign_agreement": sign_agreement,
        "metric_names": metric_names,
        "pc_names": [f"PC{i+1}" for i in range(n_pcs)],
    }


def print_cross_seed_report(stability: dict):
    """Print cross-seed stability report."""
    print("\n" + "=" * 70)
    print(f"CROSS-SEED STABILITY ({stability['n_seeds']} seeds: {stability['seeds']})")
    print("=" * 70)

    print("\nExplained variance ratio (mean ± std):")
    for i, (m, s) in enumerate(zip(stability["evr_mean"], stability["evr_std"])):
        print(f"  PC{i+1}: {m:.4f} ± {s:.4f}  ({m*100:.1f}% ± {s*100:.1f}%)")

    total_mean = stability["evr_mean"].sum()
    total_std = stability["evr_std"].sum()
    print(f"  Total: {total_mean:.4f} ± {total_std:.4f}")

    sa = stability["sign_agreement"]
    full_agree = (sa == 1.0).sum()
    partial = (sa == 0.5).sum()
    total = sa.size
    print(f"\nSign agreement: {full_agree}/{total} ({full_agree/total:.0%}) fully consistent")
    if partial > 0:
        print(f"                {partial}/{total} ({partial/total:.0%}) partially consistent")

    print("\nPer-PC sign consistency (fraction of metrics with consistent direction):")
    for pi in range(len(stability["pc_names"])):
        consistent = np.sum(sa[pi] >= 0.5) / len(stability["metric_names"])
        print(f"  {stability['pc_names'][pi]}: {consistent:.0%}")


# ---------------------------------------------------------------------------
# 6. Generate evidence JSON for thesis
# ---------------------------------------------------------------------------

def generate_evidence_json(
    mono: dict,
    cross: dict,
    effects: dict,
    stability: dict | None = None,
    save_path: str | None = None,
) -> dict:
    """Compile all validation evidence into a single JSON."""
    evidence = {
        "monotonicity": {
            "description": "Spearman rank correlation between control value and signal metric",
            "strong_pairs": [],
        },
        "orthogonality": cross,
        "effect_sizes": effects,
    }

    for pi, pc in enumerate(mono["pc_names"]):
        for mi, mn in enumerate(mono["metric_names"]):
            r = mono["rho"][pi, mi]
            p = mono["pvalue"][pi, mi]
            if abs(r) > 0.7 and p < 0.05:
                evidence["monotonicity"]["strong_pairs"].append({
                    "pc": pc, "metric": mn,
                    "rho": round(float(r), 4),
                    "p_value": round(float(p), 6),
                })

    if stability:
        evidence["cross_seed_stability"] = {
            "n_seeds": stability["n_seeds"],
            "seeds": stability["seeds"],
            "evr_mean": [round(float(x), 4) for x in stability["evr_mean"]],
            "evr_std": [round(float(x), 4) for x in stability["evr_std"]],
            "sign_consistency_per_pc": [
                round(float(np.sum(stability["sign_agreement"][pi] >= 0.5) / len(stability["metric_names"])), 2)
                for pi in range(len(stability["pc_names"]))
            ],
        }

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(evidence, f, indent=2, default=str)
        print(f"  Saved: {p}")

    return evidence
