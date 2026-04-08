"""Statistical validation of PC control dimensions.

Produces three categories of evidence:
  1. Monotonicity: Spearman ρ for each (PC, metric) pair
  2. Orthogonality: cross-influence matrix showing each PC primarily
     affects its bound metrics, not others'
  3. Effect size: Cohen's d or relative change for PC1-4 vs PC5-8
"""

from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.pipelines.pca_control import get_component_matrix, get_explained_variance_ratio


# ---------------------------------------------------------------------------
# 1. Monotonicity: Spearman correlation matrix
# ---------------------------------------------------------------------------

def compute_monotonicity_matrix(sweep_results: list[dict]) -> dict:
    """Compute Spearman ρ and p-value for each (PC, metric) pair.

    Args:
        sweep_results: list of dicts from sweep_axis(with_metrics=True), one per PC.
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


def choose_anchor_indices(
    Z_scores: np.ndarray,
    n_anchors: int,
    seed: int = 42,
    focus_dims: int = 3,
) -> list[int]:
    """Choose central-to-moderate real samples for multi-anchor sweeps."""
    scores = np.linalg.norm(Z_scores[:, : min(focus_dims, Z_scores.shape[1])], axis=1)
    out: list[int] = []

    for q in np.linspace(0.2, 0.8, n_anchors):
        idx = int(np.argmin(np.abs(scores - np.quantile(scores, q))))
        if idx not in out:
            out.append(idx)

    rng = np.random.default_rng(seed)
    while len(out) < n_anchors:
        idx = int(rng.integers(0, Z_scores.shape[0]))
        if idx not in out:
            out.append(idx)

    return out[:n_anchors]


def safe_spearman(x, y) -> tuple[float, float]:
    """Compute Spearman correlation while handling flat metric traces."""
    y = np.asarray(y, dtype=float)
    if len(y) < 3 or np.std(y) < 1e-12:
        return 0.0, 1.0
    r, p = spearmanr(x, y)
    if not np.isfinite(r):
        r = 0.0
    if not np.isfinite(p):
        p = 1.0
    return float(r), float(p)


def relative_effect_size(y) -> float:
    """Return relative sweep span for one metric trace."""
    y = np.asarray(y, dtype=float)
    return float((y.max() - y.min()) / (np.mean(np.abs(y)) + 1e-10))


def summarize_multi_anchor_metrics(
    sweep_items: list[dict],
    min_active_rho: float = 0.2,
) -> dict:
    """Aggregate monotonicity/effect statistics across multiple anchor sweeps.

    Args:
        sweep_items: List of per-anchor sweep dicts for one axis. Each item must
            contain ``values`` and ``metrics`` fields from ``sweep_axis``.
        min_active_rho: Minimum |rho| to count a sweep toward sign consistency.

    Returns:
        Dict with per-metric aggregated stats and a score-ranked metric list.
    """
    metric_names = list(sweep_items[0]["metrics"][0].keys())
    per_metric = {}

    for metric_name in metric_names:
        rhos = []
        pvals = []
        effects = []

        for item in sweep_items:
            values = np.asarray(item["values"], dtype=float)
            ys = np.asarray([m[metric_name] for m in item["metrics"]], dtype=float)
            rho, pval = safe_spearman(values, ys)
            rhos.append(rho)
            pvals.append(pval)
            effects.append(relative_effect_size(ys))

        active = [np.sign(rho) for rho in rhos if abs(rho) >= min_active_rho]
        if active:
            sign_consistency = max(sum(sign > 0 for sign in active), sum(sign < 0 for sign in active)) / len(active)
        else:
            sign_consistency = 0.0

        mean_abs_rho = float(np.mean(np.abs(rhos)))
        mean_effect = float(np.mean(effects))
        per_metric[metric_name] = {
            "mean_rho": float(np.mean(rhos)),
            "mean_abs_rho": mean_abs_rho,
            "mean_pvalue": float(np.mean(pvals)),
            "mean_effect": mean_effect,
            "sign_consistency": float(sign_consistency),
            "score": mean_abs_rho * min(mean_effect, 2.0) * max(sign_consistency, 0.25),
        }

    ranking = sorted(per_metric.items(), key=lambda kv: kv[1]["score"], reverse=True)
    return {
        "metric_names": metric_names,
        "per_metric": per_metric,
        "ranking": ranking,
    }


def build_mean_monotonicity_matrix(pc_summaries: dict[str, dict]) -> dict:
    """Build a monotonicity-like matrix from aggregated multi-anchor summaries."""
    pc_names = list(pc_summaries.keys())
    metric_names = pc_summaries[pc_names[0]]["metric_names"]

    rho = np.zeros((len(pc_names), len(metric_names)))
    pval = np.zeros((len(pc_names), len(metric_names)))

    for pi, pc in enumerate(pc_names):
        for mi, metric_name in enumerate(metric_names):
            info = pc_summaries[pc]["per_metric"][metric_name]
            rho[pi, mi] = info["mean_rho"]
            pval[pi, mi] = info["mean_pvalue"]

    return {
        "metric_names": metric_names,
        "pc_names": pc_names,
        "rho": rho,
        "pvalue": pval,
    }


def _classify_axis_quality(top_info: dict, selectivity: float, seed_alignment: float | None = None) -> str:
    """Map candidate-axis metrics to a simple strong/moderate/weak label."""
    strong = (
        top_info["mean_abs_rho"] >= 0.70
        and top_info["mean_effect"] >= 0.20
        and top_info["sign_consistency"] >= 0.80
        and selectivity >= 2.0
        and (seed_alignment is None or seed_alignment >= 0.70)
    )
    moderate = (
        top_info["mean_abs_rho"] >= 0.50
        and top_info["mean_effect"] >= 0.12
        and top_info["sign_consistency"] >= 0.65
        and selectivity >= 1.5
        and (seed_alignment is None or seed_alignment >= 0.55)
    )
    if strong:
        return "strong"
    if moderate:
        return "moderate"
    return "weak"


def summarize_candidate_axes(
    pc_summaries: dict[str, dict],
    explained_variance_ratio: np.ndarray,
    cross_seed_alignment: list[float] | None = None,
    exclude_metrics: tuple[str, ...] = ("peak_amplitude",),
    top_k_bindings: int = 2,
) -> dict:
    """Turn multi-anchor metric summaries into per-axis candidate recommendations."""
    mono = build_mean_monotonicity_matrix(pc_summaries)

    bindings = {}
    for pc in mono["pc_names"]:
        ranking = pc_summaries[pc]["ranking"]
        selected = [name for name, _ in ranking if name not in exclude_metrics][:top_k_bindings]
        if not selected and ranking:
            selected = [ranking[0][0]]
        bindings[pc] = selected

    cross = compute_cross_influence(mono, bindings)

    signature = np.array(
        [
            [pc_summaries[pc]["per_metric"][metric_name]["mean_rho"] for metric_name in mono["metric_names"]]
            for pc in mono["pc_names"]
        ],
        dtype=float,
    )
    signature_norm = signature / (np.linalg.norm(signature, axis=1, keepdims=True) + 1e-9)
    similarity = np.abs(signature_norm @ signature_norm.T)

    rows = []
    for pi, pc in enumerate(mono["pc_names"]):
        ranking = pc_summaries[pc]["ranking"]
        usable = [(name, info) for name, info in ranking if name not in exclude_metrics]
        top_name, top_info = usable[0] if usable else ranking[0]

        seed_alignment = None if cross_seed_alignment is None else float(cross_seed_alignment[pi])
        selectivity = float(cross[pc]["selectivity"])
        selectivity_term = max(min(selectivity / 2.0, 2.0), 0.25)
        seed_term = 1.0 if seed_alignment is None else max(seed_alignment, 0.25)
        overall_score = float(top_info["score"] * selectivity_term * seed_term)
        confidence = _classify_axis_quality(top_info, selectivity, seed_alignment)

        closest_idx = -1
        closest_sim = 0.0
        if len(mono["pc_names"]) > 1:
            sims = similarity[pi].copy()
            sims[pi] = -1.0
            closest_idx = int(np.argmax(sims))
            closest_sim = float(sims[closest_idx])

        dominant = "; ".join(
            [
                f"{name}:{info['mean_rho']:+.2f}/eff{info['mean_effect']:.2f}/cons{info['sign_consistency']:.2f}"
                for name, info in ranking[:4]
            ]
        )

        rows.append(
            {
                "pc": pc,
                "axis": pi,
                "explained_variance_ratio": round(float(explained_variance_ratio[pi]), 6),
                "top_metric": top_name,
                "dominant_metrics": dominant,
                "monotonicity_abs_rho": round(float(top_info["mean_abs_rho"]), 4),
                "effect_size_relative": round(float(top_info["mean_effect"]), 4),
                "cross_anchor_consistency": round(float(top_info["sign_consistency"]), 4),
                "selectivity": round(selectivity, 4),
                "cross_seed_alignment": None if seed_alignment is None else round(seed_alignment, 4),
                "closest_axis": "" if closest_idx < 0 else mono["pc_names"][closest_idx],
                "closest_axis_similarity": round(closest_sim, 4),
                "overall_score": round(overall_score, 6),
                "confidence": confidence,
                "use_recommendation": "use" if confidence == "strong" else ("review" if confidence == "moderate" else "disable"),
            }
        )

    return {
        "rows": rows,
        "bindings": bindings,
        "cross_influence": cross,
        "monotonicity": {
            "metric_names": mono["metric_names"],
            "pc_names": mono["pc_names"],
            "rho": mono["rho"].tolist(),
            "pvalue": mono["pvalue"].tolist(),
        },
        "signature_similarity": {
            "pc_names": mono["pc_names"],
            "matrix": similarity.tolist(),
        },
    }


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
# 4. Plot: selectivity bar chart
# ---------------------------------------------------------------------------

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


def compute_axis_alignment(models_by_seed: dict, n_components: int | None = None) -> dict:
    """Compute cosine alignment for any PCA-like axis set across seeds.

    This works for both baseline PCA pipelines and rotated Varimax axes.
    """
    seeds = list(models_by_seed.keys())
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds to compute cross-seed alignment")

    components = {}
    evrs = {}
    for seed, model in models_by_seed.items():
        comp = get_component_matrix(model)
        evr = get_explained_variance_ratio(model)
        n_used = comp.shape[0] if n_components is None else min(n_components, comp.shape[0])
        components[seed] = comp[:n_used]
        evrs[seed] = evr[:n_used]

    n_used = components[seeds[0]].shape[0]
    pair_alignments = {}

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            s1, s2 = seeds[i], seeds[j]
            cos_sim = np.abs(components[s1] @ components[s2].T)
            best_match = np.max(cos_sim, axis=1)
            best_idx = np.argmax(cos_sim, axis=1)

            pair_alignments[f"{s1}_vs_{s2}"] = {
                "cosine_similarity_matrix": cos_sim.tolist(),
                "best_match_per_pc": [
                    {
                        "pc": f"PC{k+1}",
                        "best_match": f"PC{best_idx[k]+1}",
                        "cosine_sim": round(float(best_match[k]), 4),
                    }
                    for k in range(n_used)
                ],
                "mean_alignment": round(float(np.mean(best_match)), 4),
            }

    per_pc_avg = []
    for k in range(n_used):
        sims = [pair_data["best_match_per_pc"][k]["cosine_sim"] for pair_data in pair_alignments.values()]
        per_pc_avg.append(round(float(np.mean(sims)), 4))

    return {
        "seeds": seeds,
        "pair_alignments": pair_alignments,
        "per_pc_avg_alignment": per_pc_avg,
        "overall_mean_alignment": round(float(np.mean(per_pc_avg)), 4),
        "evr_per_seed": {seed: [round(float(v), 4) for v in evrs[seed]] for seed in seeds},
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
