"""Validate PC control dimensions with statistical evidence.

Produces:
  1. Monotonicity heatmap (Spearman ρ for each PC × Metric)
  2. Cross-influence / selectivity analysis
  3. Effect size comparison (PC1-4 vs PC5-8)
  4. Evidence JSON for thesis

Usage:
    python scripts/validate_controls.py \
        --config configs/vae_balanced.yaml \
        --data_dir /path/to/wavs \
        --checkpoint outputs/vae_balanced/best_model.pt \
        --output_dir outputs/validation
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.preprocessing import collect_clean_wavs, estimate_global_rms
from src.data.dataset import HapticWavDataset
from src.models.conv_vae import ConvVAE
from src.pipelines.latent_extraction import extract_latent_vectors
from src.pipelines.pca_control import fit_pca_pipeline
from src.pipelines.control_spec import (
    compute_control_ranges,
    sweep_with_metrics,
)
from src.eval.pc_validation import (
    compute_monotonicity_matrix,
    print_monotonicity_report,
    compute_cross_influence,
    print_cross_influence_report,
    compute_effect_sizes,
    print_effect_size_report,
    plot_monotonicity_heatmap,
    plot_selectivity_bar,
    generate_evidence_json,
)

# Metric bindings: which metrics each PC is expected to primarily control.
# These will be validated (not assumed) by the monotonicity analysis.
CANDIDATE_BINDINGS = {
    "PC1": ["rms_energy", "peak_amplitude"],
    "PC2": ["envelope_decay_slope_dBps", "late_early_energy_ratio"],
    "PC3": ["spectral_centroid_hz", "low_high_band_ratio"],
    "PC4": ["attack_time_s", "crest_factor"],
    "PC5": ["onset_density_ps", "ioi_entropy_bits"],
    "PC6": ["onset_density_ps", "gap_ratio"],
    "PC7": ["zero_crossing_rate_ps", "gap_ratio"],
    "PC8": ["short_term_variance", "am_modulation_index"],
}


def load_model_and_data(config, args, device):
    """Load model, data, and PCA pipeline."""
    data_cfg = config["data"]
    wav_files = collect_clean_wavs(args.data_dir)
    assert len(wav_files) > 0

    N = len(wav_files)
    perm = np.random.permutation(N)
    train_files = [wav_files[i] for i in perm[:int(data_cfg["train_split"] * N)]]
    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

    all_ds = HapticWavDataset(
        wav_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
    )
    all_loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)

    model_cfg = config["model"]
    model = ConvVAE(
        T=data_cfg["T"],
        latent_dim=model_cfg["latent_dim"],
        channels=tuple(model_cfg["channels"]),
        first_kernel=model_cfg.get("first_kernel", 25),
        kernel_size=model_cfg.get("kernel_size", 9),
        activation=model_cfg.get("activation", "leaky_relu"),
        norm=model_cfg.get("norm", "group"),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    return model, all_loader, data_cfg


def main():
    parser = argparse.ArgumentParser(description="Validate PC control dimensions")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/validation")
    parser.add_argument("--pca_dir", type=str, default=None)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_sweep_steps", type=int, default=21,
                        help="More steps = more reliable Spearman ρ (default: 21)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, all_loader, data_cfg = load_model_and_data(config, args, device)

    # --- PCA ---
    if args.pca_dir and os.path.exists(os.path.join(args.pca_dir, "pca_pipe.pkl")):
        print(f"📦 Loading existing PCA from {args.pca_dir}")
        with open(os.path.join(args.pca_dir, "pca_pipe.pkl"), "rb") as f:
            pipe = pickle.load(f)
        Z_pca = np.load(os.path.join(args.pca_dir, "Z_pca.npy"))
    else:
        Z = extract_latent_vectors(model, all_loader, device)
        pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components,
                                       save_dir=args.output_dir)

    evr = pipe.named_steps["pca"].explained_variance_ratio_

    # --- Compute control ranges (P5-P95) ---
    ranges = compute_control_ranges(Z_pca)

    # --- Run sweeps with all metrics ---
    print("\n" + "=" * 60)
    print(f"Running {args.n_components} PC sweeps × {args.n_sweep_steps} steps × 15 metrics")
    print("=" * 60)

    sweep_results = []
    for i in range(args.n_components):
        r = ranges[i]
        print(f"  PC{i+1}: [{r['p5']:+.2f}, {r['p95']:+.2f}]")
        sweep = sweep_with_metrics(
            pipe, model, device,
            axis=i,
            sweep_range=(r["p5"], r["p95"]),
            n_steps=args.n_sweep_steps,
            T=data_cfg["T"], sr=data_cfg["sr"],
        )
        sweep_results.append(sweep)

    # --- 1. Monotonicity ---
    mono = compute_monotonicity_matrix(sweep_results)
    print_monotonicity_report(mono)
    plot_monotonicity_heatmap(mono, save_path=os.path.join(args.output_dir, "monotonicity_heatmap.png"))

    # --- 2. Orthogonality ---
    # First, auto-detect bindings from data: for each PC, pick top-2 metrics by |ρ|
    auto_bindings = {}
    for pi, pc in enumerate(mono["pc_names"]):
        rho_abs = np.abs(mono["rho"][pi])
        top2_idx = np.argsort(rho_abs)[-2:][::-1]
        auto_bindings[pc] = [mono["metric_names"][idx] for idx in top2_idx]
        print(f"  Auto-binding {pc}: {auto_bindings[pc]}")

    cross = compute_cross_influence(mono, auto_bindings)
    print_cross_influence_report(cross)
    plot_selectivity_bar(cross, save_path=os.path.join(args.output_dir, "selectivity.png"))

    # --- 3. Effect sizes ---
    effects = compute_effect_sizes(sweep_results)
    print_effect_size_report(effects)

    # --- Save evidence ---
    evidence = generate_evidence_json(
        mono, cross, effects,
        save_path=os.path.join(args.output_dir, "validation_evidence.json"),
    )

    # --- Print data-driven bindings summary ---
    print("\n" + "=" * 60)
    print("DATA-DRIVEN METRIC BINDINGS (for controls_table.md)")
    print("=" * 60)
    for pi, pc in enumerate(mono["pc_names"]):
        rho_row = mono["rho"][pi]
        pval_row = mono["pvalue"][pi]
        var_pct = evr[pi] * 100

        sig_pairs = []
        for mi, mn in enumerate(mono["metric_names"]):
            if abs(rho_row[mi]) > 0.7 and pval_row[mi] < 0.05:
                direction = "↑" if rho_row[mi] > 0 else "↓"
                sig_pairs.append((mn, rho_row[mi], direction))

        sig_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  {pc} ({var_pct:.1f}% var):")
        if sig_pairs:
            for mn, rho, d in sig_pairs[:4]:
                print(f"    {d} {mn}: ρ={rho:+.3f}")
        else:
            print("    No strong monotonic metrics (|ρ| > 0.7)")

    print(f"\n🏁 Validation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
