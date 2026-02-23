"""Extract latent vectors from trained VAE and fit PCA control dimensions.

Usage:
    python scripts/extract_and_pca.py \
        --config configs/vae_default.yaml \
        --data_dir /path/to/wavs \
        --checkpoint outputs/vae_default/best_model.pt \
        --output_dir outputs/pca \
        --n_components 8
"""

import argparse
import os
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
from src.pipelines.pca_control import fit_pca_pipeline, single_axis_sweep, plot_sweep


def main():
    parser = argparse.ArgumentParser(description="Extract latents and fit PCA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/pca")
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--sweep", action="store_true", help="Run single-axis sweeps for all PCs")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    data_cfg = config["data"]
    wav_files = collect_clean_wavs(args.data_dir)
    assert len(wav_files) > 0, f"No WAV files found in {args.data_dir}"

    N = len(wav_files)
    perm = np.random.permutation(N)
    train_files = [wav_files[i] for i in perm[:int(data_cfg["train_split"] * N)]]
    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

    # Use ALL files for latent extraction (train + val)
    all_ds = HapticWavDataset(
        wav_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
    )
    all_loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")
    print(f"   Latent dim: {model.latent_dim}, Dataset size: {len(all_ds)}")

    # --- Step 1: Extract latent vectors ---
    print("\n" + "=" * 60)
    print("STEP 1: Extracting latent vectors (mu)")
    print("=" * 60)
    Z = extract_latent_vectors(model, all_loader, device)
    z_path = os.path.join(args.output_dir, "Z.npy")
    np.save(z_path, Z)
    print(f"  Saved: {z_path}")

    # --- Step 2: Fit PCA ---
    print("\n" + "=" * 60)
    print(f"STEP 2: PCA ({Z.shape[1]}D → {args.n_components}D)")
    print("=" * 60)
    pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components, save_dir=args.output_dir)

    # --- Step 4: Single-axis sweeps ---
    if args.sweep:
        print("\n" + "=" * 60)
        print("STEP 4: Single-axis sweeps")
        print("=" * 60)
        for ax in range(args.n_components):
            print(f"\n  Sweeping PC{ax+1}...")
            result = single_axis_sweep(
                pipe, model, device,
                axis=ax, sweep_range=(-2.0, 2.0), n_steps=9,
                T=data_cfg["T"],
            )
            plot_sweep(result, sr=data_cfg["sr"],
                       save_path=os.path.join(args.output_dir, f"sweep_PC{ax+1}.png"))

    print(f"\n🏁 All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
