"""CLI entry point for training haptic signal models.

Usage:
    python scripts/train.py --config configs/vae_default.yaml --data_dir /path/to/wavs --output_dir outputs
"""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.preprocessing import collect_clean_wavs, estimate_global_rms
from src.data.dataset import HapticWavDataset
from src.models.conv_vae import ConvVAE
from src.models.conv_ae import ConvAE
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train haptic signal model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for checkpoints and logs")
    args = parser.parse_args()

    config = load_config(args.config)
    config["output_dir"] = args.output_dir

    set_seed(config.get("seed", 42))

    # --- Data ---
    data_cfg = config["data"]
    print(f"📂 Collecting WAV files from: {args.data_dir}")
    wav_files = collect_clean_wavs(args.data_dir)
    assert len(wav_files) > 0, f"No WAV files found in {args.data_dir}"
    print(f"   Found {len(wav_files)} clean WAV files")

    N = len(wav_files)
    perm = np.random.permutation(N)
    split = int(data_cfg["train_split"] * N)
    train_files = [wav_files[i] for i in perm[:split]]
    val_files = [wav_files[i] for i in perm[split:]]
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}")

    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])
    print(f"   Global RMS: {global_rms:.6f}")

    T = data_cfg["T"]
    batch_size = config["training"]["batch_size"]

    train_ds = HapticWavDataset(
        train_files, T=T, sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
        use_minmax=data_cfg["use_minmax"],
    )
    val_ds = HapticWavDataset(
        val_files, T=T, sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
        use_minmax=data_cfg["use_minmax"],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"   Batches: train={len(train_loader)}, val={len(val_loader)}")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    model_cfg = config["model"]
    model_type = config.get("model_type", "vae")

    if model_type == "vae":
        model = ConvVAE(
            T=T,
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
            first_kernel=model_cfg.get("first_kernel", 25),
            kernel_size=model_cfg.get("kernel_size", 9),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
            logvar_clip=tuple(model_cfg.get("logvar_clip", [-10, 10])),
        )
    else:
        model = ConvAE(
            T=T,
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
            kernel_size=model_cfg.get("kernel_size", 9),
            use_batchnorm=model_cfg.get("norm", "batchnorm") == "batchnorm",
        )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: {model_type.upper()}, params: {total_params:,}")

    # --- Train ---
    trainer = Trainer(model, config, device)
    results = trainer.train(train_loader, val_loader)

    print(f"\n🏁 Training complete. Results in: {results['run_dir']}")


if __name__ == "__main__":
    main()
