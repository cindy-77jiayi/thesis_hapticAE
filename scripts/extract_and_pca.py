"""Extract latent vectors from trained VAE and fit PCA control dimensions.

Usage:
    python scripts/extract_and_pca.py \
        --config configs/vae_default.yaml \
        --data_dir /path/to/WavCaps \
        --checkpoint outputs/vae_default/best_model.pt \
        --output_dir outputs/pca \
        --n_components 8
"""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders import build_dataloaders, build_model, load_checkpoint
from src.pipelines.latent_extraction import extract_latent_vectors
from src.pipelines.pca_control import fit_pca_pipeline, sweep_axis, plot_sweep


def main():
    parser = argparse.ArgumentParser(description="Extract latents and fit PCA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True, help="Prepared haptic dataset directory")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/pca")
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--sweep", action="store_true", help="Run single-axis sweeps for all PCs")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    data_cfg = config["data"]

    # --- Data (full dataset for extraction) ---
    analysis_batch_size = int(config["data"].get("analysis_batch_size", 4))
    data = build_dataloaders(config, args.data_dir, batch_size=analysis_batch_size, full_dataset=True)
    print(f"   Dataset size: {len(data['audio_files'])}")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # --- Step 1: Extract latent vectors ---
    print("\n" + "=" * 60)
    print("STEP 1: Extracting latent vectors (mu)")
    print("=" * 60)
    Z = extract_latent_vectors(model, data["all_loader"], device)
    z_path = os.path.join(args.output_dir, "Z.npy")
    np.save(z_path, Z)
    print(f"  Saved: {z_path}")

    # --- Step 2: Fit PCA ---
    print("\n" + "=" * 60)
    print(f"STEP 2: PCA ({Z.shape[1]}D → {args.n_components}D)")
    print("=" * 60)
    pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components, save_dir=args.output_dir)

    # --- Step 3: Single-axis sweeps ---
    if args.sweep:
        print("\n" + "=" * 60)
        print("STEP 3: Single-axis sweeps")
        print("=" * 60)
        for ax in range(args.n_components):
            print(f"\n  Sweeping PC{ax+1}...")
            result = sweep_axis(
                pipe, model, device,
                axis=ax, sweep_range=(-2.0, 2.0), n_steps=9,
                T=data_cfg["T"],
            )
            plot_sweep(result, sr=data_cfg["sr"],
                       save_path=os.path.join(args.output_dir, f"sweep_PC{ax+1}.png"))

    print(f"\n🏁 All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
