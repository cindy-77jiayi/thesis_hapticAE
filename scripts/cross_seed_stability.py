"""Cross-seed stability test for PC control dimensions.

Trains multiple seeds, extracts PCA, and compares structure stability.

Usage (run 3 times with different configs, then compare):

  # Step 1: Train each seed (or reuse existing checkpoints)
  python scripts/train.py --config configs/vae_balanced.yaml --data_dir DATA --output_dir outputs
  python scripts/train.py --config configs/vae_balanced_s123.yaml --data_dir DATA --output_dir outputs
  python scripts/train.py --config configs/vae_balanced_s456.yaml --data_dir DATA --output_dir outputs

  # Step 2: Run this script to compare
  python scripts/cross_seed_stability.py \
      --configs configs/vae_balanced.yaml configs/vae_balanced_s123.yaml configs/vae_balanced_s456.yaml \
      --data_dir /path/to/wavs \
      --output_base outputs \
      --output_dir outputs/cross_seed
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
from src.pipelines.control_spec import compute_control_ranges, sweep_with_metrics
from src.eval.pc_validation import (
    compute_monotonicity_matrix,
    compare_cross_seed,
    print_cross_seed_report,
    generate_evidence_json,
)


def main():
    parser = argparse.ArgumentParser(description="Cross-seed stability test")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="List of config files (one per seed)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_base", type=str, default="outputs",
                        help="Base dir where training outputs are stored")
    parser.add_argument("--output_dir", type=str, default="outputs/cross_seed")
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_sweep_steps", type=int, default=21)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_results = []

    for cfg_path in args.configs:
        config = load_config(cfg_path)
        seed = config.get("seed", 42)
        run_name = config.get("run_name", "unknown")
        set_seed(seed)

        print(f"\n{'='*60}")
        print(f"Processing seed={seed}, run={run_name}")
        print(f"{'='*60}")

        # --- Data ---
        data_cfg = config["data"]
        wav_files = collect_clean_wavs(args.data_dir)
        N = len(wav_files)
        perm = np.random.permutation(N)
        train_files = [wav_files[i] for i in perm[:int(data_cfg["train_split"] * N)]]
        global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

        all_ds = HapticWavDataset(
            wav_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
            global_rms=global_rms, scale=data_cfg["scale"],
        )
        all_loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)

        # --- Model ---
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

        ckpt_path = os.path.join(args.output_base, run_name, "best_model.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        print(f"  ✅ Loaded: {ckpt_path}")

        # --- Extract + PCA ---
        Z = extract_latent_vectors(model, all_loader, device)
        pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components)
        evr = pipe.named_steps["pca"].explained_variance_ratio_

        # --- Sweep + monotonicity ---
        ranges = compute_control_ranges(Z_pca)
        sweep_results = []
        for i in range(args.n_components):
            r = ranges[i]
            sweep = sweep_with_metrics(
                pipe, model, device,
                axis=i,
                sweep_range=(r["p5"], r["p95"]),
                n_steps=args.n_sweep_steps,
                T=data_cfg["T"], sr=data_cfg["sr"],
            )
            sweep_results.append(sweep)

        mono = compute_monotonicity_matrix(sweep_results)

        seed_results.append({
            "seed": seed,
            "run_name": run_name,
            "explained_variance_ratio": evr,
            "mono": mono,
        })

    # --- Compare ---
    stability = compare_cross_seed(seed_results)
    print_cross_seed_report(stability)

    # --- Plot comparison ---
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for sr in seed_results:
        cumvar = np.cumsum(sr["explained_variance_ratio"])
        axes[0].plot(range(1, len(cumvar)+1), cumvar, "o-", label=f"seed={sr['seed']}")
    axes[0].set_xlabel("Number of PCs")
    axes[0].set_ylabel("Cumulative explained variance")
    axes[0].set_title("Explained variance stability across seeds")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    pcs = [f"PC{i+1}" for i in range(args.n_components)]
    x = np.arange(args.n_components)
    for si, sr in enumerate(seed_results):
        offset = (si - len(seed_results)/2) * 0.2
        axes[1].bar(x + offset, sr["explained_variance_ratio"] * 100,
                     width=0.2, label=f"seed={sr['seed']}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pcs)
    axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_title("Per-PC variance across seeds")
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(args.output_dir, "cross_seed_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")
    plt.show()

    # --- Save stability report ---
    import json
    report = {
        "n_seeds": stability["n_seeds"],
        "seeds": stability["seeds"],
        "evr_mean": [round(float(x), 4) for x in stability["evr_mean"]],
        "evr_std": [round(float(x), 4) for x in stability["evr_std"]],
        "total_var_mean": round(float(stability["evr_mean"].sum() * 100), 1),
        "sign_consistency_per_pc": [
            round(float(np.sum(stability["sign_agreement"][pi] >= 0.5) / len(stability["metric_names"])), 2)
            for pi in range(args.n_components)
        ],
    }
    report_path = os.path.join(args.output_dir, "cross_seed_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    print(f"\n🏁 Cross-seed results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
