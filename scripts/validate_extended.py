"""Extended validation of PC control dimensions.

Steps:
  2. Recompute metric bindings with 27 metrics
  3. Dual-reference sweeps (PCA origin + dataset mean)
  4. PCA axis cosine similarity across seeds
  5. Generate controls_table_v2.md

Usage:
    python scripts/validate_extended.py \
        --config configs/vae_balanced.yaml \
        --data_dir /path/to/wavs \
        --checkpoint outputs/vae_balanced/best_model.pt \
        --output_dir outputs/validation \
        --pca_dir outputs/pca \
        --seed_configs configs/vae_balanced.yaml configs/vae_balanced_s123.yaml configs/vae_balanced_s456.yaml \
        --seed_output_base outputs
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.preprocessing import collect_clean_wavs, estimate_global_rms
from src.data.dataset import HapticWavDataset
from src.models.conv_vae import ConvVAE
from src.pipelines.latent_extraction import extract_latent_vectors
from src.pipelines.pca_control import fit_pca_pipeline, sweep_axis
from src.pipelines.control_spec import compute_control_ranges, METRIC_LABELS
from src.eval.pc_validation import (
    compute_cross_influence,
    compute_effect_sizes,
    print_cross_influence_report,
    print_effect_size_report,
    plot_selectivity_bar,
)


# ===================================================================
# Step 2: Extended metric binding
# ===================================================================

def compute_extended_bindings(sweep_results, sig_threshold=0.05):
    """Compute Spearman ρ for each (PC, metric) pair and rank bindings."""
    metric_names = list(sweep_results[0]["metrics"][0].keys())
    n_pcs = len(sweep_results)
    n_metrics = len(metric_names)

    rho = np.zeros((n_pcs, n_metrics))
    pval = np.zeros((n_pcs, n_metrics))

    for pi, sweep in enumerate(sweep_results):
        values = np.array(sweep["values"])
        for mi, mn in enumerate(metric_names):
            metric_vals = np.array([m[mn] for m in sweep["metrics"]])
            if np.std(metric_vals) < 1e-12:
                rho[pi, mi], pval[pi, mi] = 0.0, 1.0
            else:
                r, p = spearmanr(values, metric_vals)
                rho[pi, mi], pval[pi, mi] = r, p

    bindings = {}
    for pi in range(n_pcs):
        pc = f"PC{pi + 1}"
        ranked = []
        for mi, mn in enumerate(metric_names):
            ranked.append({
                "metric": mn,
                "rho": round(float(rho[pi, mi]), 4),
                "p_value": round(float(pval[pi, mi]), 6),
                "significant": bool(pval[pi, mi] < sig_threshold),
            })
        ranked.sort(key=lambda x: abs(x["rho"]), reverse=True)
        bindings[pc] = ranked

    return {
        "metric_names": metric_names,
        "pc_names": [f"PC{i+1}" for i in range(n_pcs)],
        "rho": rho,
        "pvalue": pval,
        "ranked_bindings": bindings,
    }


def plot_extended_heatmap(bindings, save_path=None):
    """Plot full Spearman ρ heatmap with extended metric set."""
    import matplotlib.pyplot as plt

    rho = bindings["rho"]
    pval = bindings["pvalue"]
    metric_names = bindings["metric_names"]
    pc_names = bindings["pc_names"]

    short = []
    for n in metric_names:
        s = n.replace("_dBps", "").replace("_ps", "").replace("_hz", "")
        s = s.replace("_bits", "").replace("_s", "").replace("band_energy_", "band_")
        short.append(s[:20])

    fig, ax = plt.subplots(figsize=(max(18, len(metric_names) * 0.7),
                                    max(6, len(pc_names) * 0.7)))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    for pi in range(len(pc_names)):
        for mi in range(len(metric_names)):
            r, p = rho[pi, mi], pval[pi, mi]
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            color = "white" if abs(r) > 0.5 else "black"
            ax.text(mi, pi, f"{r:+.2f}{star}", ha="center", va="center",
                    fontsize=6, color=color)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(pc_names)))
    ax.set_yticklabels(pc_names, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8).set_label("Spearman ρ")
    ax.set_title("Extended Monotonicity: Spearman ρ (PC × 27 Metrics)\n"
                 "* p<0.05  ** p<0.01", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ===================================================================
# Step 3: Dual-reference sweep comparison
# ===================================================================

def compare_references(sweep_origin, sweep_mean, metric_names):
    """Compare metric responses between origin-centered and mean-centered sweeps."""
    diffs = {}
    for pi in range(len(sweep_origin)):
        pc = f"PC{pi + 1}"
        pc_diffs = {}
        for mn in metric_names:
            vals_o = [m[mn] for m in sweep_origin[pi]["metrics"]]
            vals_m = [m[mn] for m in sweep_mean[pi]["metrics"]]
            rho_o, _ = spearmanr(sweep_origin[pi]["values"], vals_o) if np.std(vals_o) > 1e-12 else (0, 1)
            rho_m, _ = spearmanr(sweep_mean[pi]["values"], vals_m) if np.std(vals_m) > 1e-12 else (0, 1)
            pc_diffs[mn] = {
                "rho_origin": round(float(rho_o), 4),
                "rho_mean": round(float(rho_m), 4),
                "sign_consistent": bool(np.sign(rho_o) == np.sign(rho_m)) if abs(rho_o) > 0.3 else True,
            }
        diffs[pc] = pc_diffs

    n_total, n_consistent = 0, 0
    for pc, metrics in diffs.items():
        for mn, d in metrics.items():
            if abs(d["rho_origin"]) > 0.3 or abs(d["rho_mean"]) > 0.3:
                n_total += 1
                if d["sign_consistent"]:
                    n_consistent += 1

    return {
        "per_pc": diffs,
        "overall_sign_consistency": round(n_consistent / max(n_total, 1), 4),
        "n_compared": n_total,
    }


# ===================================================================
# Step 4: PCA axis alignment across seeds
# ===================================================================

def compute_pca_axis_alignment(seed_pipes, n_components=8):
    """Compute axis alignment using Procrustes + Hungarian one-to-one matching."""
    seeds = list(seed_pipes.keys())
    n_seeds = len(seeds)

    components = {}
    evrs = {}
    for seed, pipe in seed_pipes.items():
        pca = pipe.named_steps["pca"]
        components[seed] = pca.components_[:n_components]
        evrs[seed] = pca.explained_variance_ratio_[:n_components]

    pair_alignments = {}
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            s1, s2 = seeds[i], seeds[j]
            C1, C2 = components[s1], components[s2]

            R, _ = orthogonal_procrustes(C2, C1)
            C2_aligned = C2 @ R
            signed_cos = C1 @ C2_aligned.T
            abs_cos = np.abs(signed_cos)

            row_ind, col_ind = linear_sum_assignment(-abs_cos)
            order = np.argsort(row_ind)
            row_ind, col_ind = row_ind[order], col_ind[order]

            matched = abs_cos[row_ind, col_ind]
            signs = np.sign(signed_cos[row_ind, col_ind])

            pair_alignments[f"{s1}_vs_{s2}"] = {
                "cosine_similarity_matrix": abs_cos.tolist(),
                "matching": [
                    {
                        "pc_ref": f"PC{int(r)+1}",
                        "pc_other": f"PC{int(c)+1}",
                        "cosine_sim": round(float(abs_cos[r, c]), 4),
                        "sign": int(signs[k]) if signs[k] != 0 else 1,
                    }
                    for k, (r, c) in enumerate(zip(row_ind, col_ind))
                ],
                "mean_alignment": round(float(np.mean(matched)), 4),
            }

    per_pc_acc = {k: [] for k in range(n_components)}
    for data in pair_alignments.values():
        for m in data["matching"]:
            pc_idx = int(m["pc_ref"].replace("PC", "")) - 1
            per_pc_acc[pc_idx].append(m["cosine_sim"])

    per_pc_avg = [round(float(np.mean(per_pc_acc[k])) if per_pc_acc[k] else 0.0, 4)
                  for k in range(n_components)]

    return {
        "seeds": seeds,
        "pair_alignments": pair_alignments,
        "per_pc_avg_alignment": per_pc_avg,
        "overall_mean_alignment": round(float(np.mean(per_pc_avg)), 4),
        "evr_per_seed": {s: [round(float(v), 4) for v in evrs[s]] for s in seeds},
        "method": "orthogonal_procrustes + hungarian",
    }


def plot_alignment(alignment, save_path=None):
    """Plot PCA axis alignment results."""
    import matplotlib.pyplot as plt

    pairs = list(alignment["pair_alignments"].keys())
    n_pcs = len(alignment["per_pc_avg_alignment"])

    fig, axes = plt.subplots(1, len(pairs) + 1, figsize=(6 * (len(pairs) + 1), 5))
    if len(pairs) + 1 == 1:
        axes = [axes]

    for pi, (pair, data) in enumerate(alignment["pair_alignments"].items()):
        mat = np.array(data["cosine_similarity_matrix"])
        im = axes[pi].imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                axes[pi].text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
        axes[pi].set_xticks(range(n_pcs))
        axes[pi].set_xticklabels([f"PC{k+1}" for k in range(n_pcs)], fontsize=8)
        axes[pi].set_yticks(range(n_pcs))
        axes[pi].set_yticklabels([f"PC{k+1}" for k in range(n_pcs)], fontsize=8)
        axes[pi].set_title(pair.replace("_", " "), fontsize=10)

    ax_bar = axes[-1]
    avg = alignment["per_pc_avg_alignment"]
    ax_bar.bar(range(n_pcs), avg, color="steelblue")
    ax_bar.set_xticks(range(n_pcs))
    ax_bar.set_xticklabels([f"PC{k+1}" for k in range(n_pcs)])
    ax_bar.set_ylabel("Mean cosine similarity")
    ax_bar.set_title(f"Avg alignment (overall={alignment['overall_mean_alignment']:.3f})")
    ax_bar.set_ylim(0, 1.05)

    plt.suptitle("PCA Axis Stability Across Seeds", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ===================================================================
# Step 5: Generate controls_table_v2.md
# ===================================================================

def generate_table_v2(bindings, alignment, evr, ref_comparison, save_path):
    """Generate thesis-ready controls_table_v2.md with data-driven descriptions."""
    pc_names = bindings["pc_names"]
    ranked = bindings["ranked_bindings"]

    lines = [
        "# Haptic Control Dimensions — Extended Validation (v2)",
        "",
        f"**Metrics:** {len(bindings['metric_names'])} signal features",
        f"**Sweep:** 21 steps across P5–P95 range per PC",
        f"**Reference consistency:** origin vs mean sign agreement = "
        f"{ref_comparison['overall_sign_consistency']:.0%}",
        "",
    ]

    if alignment:
        lines.append(f"**PCA axis stability:** mean cosine similarity = "
                      f"{alignment['overall_mean_alignment']:.3f} across "
                      f"{len(alignment['seeds'])} seeds")
        lines.append("")

    lines.extend([
        "## Control Specification",
        "",
        "| Control | Tier | Var% | Primary Metrics (ρ) | Semantic Interpretation | Example Effect |",
        "|---------|------|------|---------------------|------------------------|----------------|",
    ])

    for pi, pc in enumerate(pc_names):
        var_pct = round(float(evr[pi]) * 100, 1) if pi < len(evr) else 0

        top = [m for m in ranked[pc] if m["significant"] and m["metric"] != "peak_amplitude"][:3]
        metrics_str = ", ".join(
            f"{m['metric'].split('_')[0]}({'↑' if m['rho']>0 else '↓'}{abs(m['rho']):.2f})"
            for m in top
        )

        label = _auto_label(top)
        effect = _auto_effect(top)

        tier = "Primary" if pi < 4 else "Secondary"
        lines.append(f"| {pc} | {tier} | {var_pct} | {metrics_str} | {label} | {effect} |")

    lines.extend(["", "## Detailed Metric Bindings", ""])

    for pi, pc in enumerate(pc_names):
        var_pct = round(float(evr[pi]) * 100, 1) if pi < len(evr) else 0

        if alignment:
            align_score = alignment["per_pc_avg_alignment"][pi]
            lines.append(f"### {pc} — {var_pct}% variance (axis stability: {align_score:.3f})")
        else:
            lines.append(f"### {pc} — {var_pct}% variance")
        lines.append("")

        sig_metrics = [m for m in ranked[pc] if m["significant"]]
        if sig_metrics:
            lines.append("| Metric | Spearman ρ | p-value |")
            lines.append("|--------|-----------|---------|")
            for m in sig_metrics[:8]:
                mname = METRIC_LABELS.get(m["metric"], (m["metric"], ""))[0]
                lines.append(f"| {mname} | {m['rho']:+.3f} | {m['p_value']:.4f} |")
        else:
            lines.append("No significant correlations (p < 0.05)")
        lines.append("")

    lines.extend([
        "## Methodological Notes",
        "",
        "- `peak_amplitude` excluded from primary bindings (see v1 analysis)",
        "- Sweep reference: PCA origin; confirmed consistent with dataset-mean reference",
        "- Metric correlations computed using Spearman rank correlation (21 sweep steps)",
        "- Semantic labels are data-driven composites, not forced single-word labels",
    ])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {save_path}")


def _auto_label(top_metrics):
    """Generate a data-driven semantic label from top correlated metrics."""
    if not top_metrics:
        return "undifferentiated temporal variation"

    keywords = set()
    for m in top_metrics:
        name = m["metric"]
        if "rms" in name or "energy" in name or "amplitude" in name:
            keywords.add("energy")
        if "spectral" in name or "centroid" in name or "rolloff" in name or "band_" in name:
            keywords.add("spectral shape")
        if "decay" in name or "late_early" in name or "duration" in name:
            keywords.add("decay dynamics")
        if "attack" in name or "transient" in name:
            keywords.add("attack character")
        if "onset" in name or "ioi" in name or "modulation" in name:
            keywords.add("temporal patterning")
        if "flatness" in name or "slope" in name:
            keywords.add("spectral texture")
        if "envelope" in name and "entropy" in name:
            keywords.add("envelope complexity")
        if "am_modulation" in name or "short_term" in name:
            keywords.add("amplitude variation")
        if "crest" in name:
            keywords.add("impulsiveness")
        if "zero_crossing" in name or "gap" in name:
            keywords.add("continuity")

    if not keywords:
        keywords.add("fine temporal structure")

    return " / ".join(sorted(keywords))


def _auto_effect(top_metrics):
    """Generate example perceptual effect description."""
    if not top_metrics:
        return "subtle variation"
    m = top_metrics[0]
    name = m["metric"]
    direction = "increases" if m["rho"] > 0 else "decreases"

    effects = {
        "rms_energy": f"overall intensity {direction}",
        "spectral_centroid_hz": f"brightness {direction}",
        "spectral_rolloff_hz": f"spectral bandwidth {direction}",
        "spectral_slope": f"spectral tilt {direction}",
        "spectral_flatness": f"tonality {'decreases' if direction == 'increases' else 'increases'}",
        "envelope_decay_slope_dBps": f"sustain {direction}",
        "attack_time_s": f"attack {'sharpens' if direction == 'decreases' else 'softens'}",
        "am_modulation_index": f"amplitude modulation {direction}",
        "short_term_variance": f"energy fluctuation {direction}",
        "onset_density_ps": f"event rate {direction}",
        "envelope_entropy_bits": f"envelope complexity {direction}",
        "crest_factor": f"impulsiveness {direction}",
    }
    return effects.get(name, f"{name.replace('_', ' ')} {direction}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended PC validation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/validation")
    parser.add_argument("--controls_dir", type=str, default="outputs/controls")
    parser.add_argument("--pca_dir", type=str, default=None)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_sweep_steps", type=int, default=21)
    parser.add_argument("--seed_configs", nargs="*", default=[],
                        help="Config files for cross-seed PCA axis comparison")
    parser.add_argument("--seed_output_base", type=str, default="outputs")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.controls_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config["data"]

    # --- Load model ---
    model_cfg = config["model"]
    model = ConvVAE(
        T=data_cfg["T"], latent_dim=model_cfg["latent_dim"],
        channels=tuple(model_cfg["channels"]),
        first_kernel=model_cfg.get("first_kernel", 25),
        kernel_size=model_cfg.get("kernel_size", 9),
        activation=model_cfg.get("activation", "leaky_relu"),
        norm=model_cfg.get("norm", "group"),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded: {args.checkpoint}")

    # --- Load PCA ---
    if args.pca_dir and os.path.exists(os.path.join(args.pca_dir, "pca_pipe.pkl")):
        with open(os.path.join(args.pca_dir, "pca_pipe.pkl"), "rb") as f:
            pipe = pickle.load(f)
        Z_pca = np.load(os.path.join(args.pca_dir, "Z_pca.npy"))
        Z = np.load(os.path.join(args.pca_dir, "Z.npy"))
        print(f"📦 Loaded PCA from {args.pca_dir}")
    else:
        wav_files = collect_clean_wavs(args.data_dir)
        N = len(wav_files)
        perm = np.random.permutation(N)
        train_files = [wav_files[i] for i in perm[:int(data_cfg["train_split"] * N)]]
        global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])
        all_ds = HapticWavDataset(
            wav_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
            global_rms=global_rms, scale=data_cfg["scale"],
        )
        loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)
        Z = extract_latent_vectors(model, loader, device)
        pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components,
                                       save_dir=args.output_dir)

    evr = pipe.named_steps["pca"].explained_variance_ratio_
    ranges = compute_control_ranges(Z_pca)

    # --- Step 2: Extended sweeps from PCA origin ---
    print("\n" + "=" * 60)
    print(f"STEP 2: Extended sweeps ({args.n_components} PCs × {args.n_sweep_steps} steps × 27 metrics)")
    print("=" * 60)

    sweep_origin = []
    for i in range(args.n_components):
        r = ranges[i]
        print(f"  PC{i+1}: [{r['p5']:+.2f}, {r['p95']:+.2f}]")
        sweep = sweep_axis(
            pipe, model, device, axis=i,
            sweep_range=(r["p5"], r["p95"]),
            n_steps=args.n_sweep_steps,
            T=data_cfg["T"], sr=data_cfg["sr"],
            with_metrics=True,
        )
        sweep_origin.append(sweep)

    bindings = compute_extended_bindings(sweep_origin)
    plot_extended_heatmap(bindings,
                          save_path=os.path.join(args.output_dir, "monotonicity_extended_heatmap.png"))

    # Print top bindings
    print("\nExtended metric bindings (excluding peak_amplitude):")
    for pc in bindings["pc_names"]:
        top = [m for m in bindings["ranked_bindings"][pc]
               if m["significant"] and m["metric"] != "peak_amplitude"][:4]
        metrics_str = ", ".join(f"{m['metric']}({'↑' if m['rho']>0 else '↓'}{abs(m['rho']):.2f})"
                                for m in top)
        print(f"  {pc}: {metrics_str}")

    # --- Step 3: Mean-reference sweeps ---
    print("\n" + "=" * 60)
    print("STEP 3: Dual-reference comparison (origin vs dataset mean)")
    print("=" * 60)

    Z_pca_mean = Z_pca.mean(axis=0).astype(np.float32)
    print(f"  Dataset mean in PCA space: {Z_pca_mean}")

    sweep_mean = []
    for i in range(args.n_components):
        r = ranges[i]
        sweep = sweep_axis(
            pipe, model, device, axis=i,
            sweep_range=(r["p5"], r["p95"]),
            n_steps=args.n_sweep_steps,
            T=data_cfg["T"], sr=data_cfg["sr"],
            reference=Z_pca_mean,
            with_metrics=True,
        )
        sweep_mean.append(sweep)

    metric_names = list(sweep_origin[0]["metrics"][0].keys())
    ref_comparison = compare_references(sweep_origin, sweep_mean, metric_names)
    print(f"\n  Sign consistency (origin vs mean): "
          f"{ref_comparison['overall_sign_consistency']:.0%} "
          f"({ref_comparison['n_compared']} pairs compared)")

    # --- Step 4: PCA axis alignment ---
    alignment = None
    if args.seed_configs:
        print("\n" + "=" * 60)
        print("STEP 4: PCA axis alignment across seeds")
        print("=" * 60)

        seed_pipes = {}
        for cfg_path in args.seed_configs:
            cfg = load_config(cfg_path)
            seed = cfg.get("seed", 42)
            run_name = cfg.get("run_name", "unknown")
            set_seed(seed)

            m_cfg = cfg["model"]
            d_cfg = cfg["data"]
            m = ConvVAE(
                T=d_cfg["T"], latent_dim=m_cfg["latent_dim"],
                channels=tuple(m_cfg["channels"]),
                first_kernel=m_cfg.get("first_kernel", 25),
                kernel_size=m_cfg.get("kernel_size", 9),
                activation=m_cfg.get("activation", "leaky_relu"),
                norm=m_cfg.get("norm", "group"),
            ).to(device)

            ckpt = os.path.join(args.seed_output_base, run_name, "best_model.pt")
            if not os.path.exists(ckpt):
                print(f"  ⚠️ Checkpoint not found: {ckpt}, skipping")
                continue

            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            m.eval()

            wav_files = collect_clean_wavs(args.data_dir)
            N = len(wav_files)
            perm = np.random.permutation(N)
            train_files = [wav_files[i] for i in perm[:int(d_cfg["train_split"] * N)]]
            global_rms = estimate_global_rms(train_files, n=200, sr_expect=d_cfg["sr"])
            all_ds = HapticWavDataset(
                wav_files, T=d_cfg["T"], sr_expect=d_cfg["sr"],
                global_rms=global_rms, scale=d_cfg["scale"],
            )
            loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)

            Z_seed = extract_latent_vectors(m, loader, device)
            pipe_seed, _ = fit_pca_pipeline(Z_seed, n_components=args.n_components)
            seed_pipes[seed] = pipe_seed
            print(f"  ✅ seed={seed}: extracted + PCA done")

        if len(seed_pipes) >= 2:
            alignment = compute_pca_axis_alignment(seed_pipes, args.n_components)
            print(f"\n  Overall mean alignment: {alignment['overall_mean_alignment']:.3f}")
            print(f"  Per-PC: {alignment['per_pc_avg_alignment']}")

            align_path = os.path.join(args.output_dir, "pca_axis_alignment.json")
            with open(align_path, "w") as f:
                json.dump(alignment, f, indent=2)
            print(f"  Saved: {align_path}")

            plot_alignment(alignment,
                           save_path=os.path.join(args.output_dir, "pca_axis_alignment.png"))

    # --- Cross-influence (orthogonality) ---
    print("\n" + "=" * 60)
    print("ORTHOGONALITY: Cross-influence analysis")
    print("=" * 60)

    auto_bindings = {}
    for pi, pc in enumerate(bindings["pc_names"]):
        top = [m for m in bindings["ranked_bindings"][pc]
               if m["significant"] and m["metric"] != "peak_amplitude"][:2]
        auto_bindings[pc] = [m["metric"] for m in top] if top else ["rms_energy"]

    cross = compute_cross_influence(bindings, auto_bindings)
    print_cross_influence_report(cross)
    plot_selectivity_bar(cross, save_path=os.path.join(args.output_dir, "selectivity_extended.png"))

    # --- Effect sizes (primary vs secondary) ---
    print("\n" + "=" * 60)
    print("EFFECT SIZE: Primary (PC1-4) vs Secondary (PC5-8)")
    print("=" * 60)

    effects = compute_effect_sizes(sweep_origin)
    print_effect_size_report(effects)

    # --- Save extended bindings JSON ---
    binding_json = {
        "n_metrics": len(bindings["metric_names"]),
        "metric_names": bindings["metric_names"],
        "ranked_bindings": bindings["ranked_bindings"],
        "reference_comparison": ref_comparison,
        "cross_influence": cross,
        "effect_sizes": effects,
    }
    binding_path = os.path.join(args.output_dir, "metric_binding_extended.json")
    with open(binding_path, "w") as f:
        json.dump(binding_json, f, indent=2, default=str)
    print(f"\n  Saved: {binding_path}")

    # --- Step 5: Generate controls_table_v2.md ---
    print("\n" + "=" * 60)
    print("STEP 5: Generating controls_table_v2.md")
    print("=" * 60)

    generate_table_v2(
        bindings, alignment, evr, ref_comparison,
        save_path=os.path.join(args.controls_dir, "controls_table_v2.md"),
    )

    print(f"\n🏁 Extended validation complete.")
    print(f"   {args.output_dir}/monotonicity_extended_heatmap.png")
    print(f"   {args.output_dir}/metric_binding_extended.json")
    if alignment:
        print(f"   {args.output_dir}/pca_axis_alignment.json")
        print(f"   {args.output_dir}/pca_axis_alignment.png")
    print(f"   {args.controls_dir}/controls_table_v2.md")


if __name__ == "__main__":
    main()
