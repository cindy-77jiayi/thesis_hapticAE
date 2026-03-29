"""CLI entry point for training haptic signal models.

Usage:
    python scripts/train.py --config configs/vae_default.yaml --data_dir /path/to/wavs --output_dir outputs
"""

import argparse
from _bootstrap import add_project_root

add_project_root()

import torch

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.loaders import build_dataloaders, build_model
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
    print(f"Collecting WAV files from: {args.data_dir}")
    data = build_dataloaders(config, args.data_dir)
    print(f"   Found {len(data['wav_files'])} clean WAV files")
    if data.get("source_counts"):
        print("   Source counts:")
        for root, count in data["source_counts"].items():
            print(f"      {root}: {count}")
    print(f"   Global RMS: {data['global_rms']:.6f}")
    seg_seconds = config["data"]["T"] / float(config["data"]["sr"])
    print(f"   Segment: T={config['data']['T']} ({seg_seconds:.3f}s @ {config['data']['sr']}Hz)")
    print(f"   Batches: train={len(data['train_loader'])}, val={len(data['val_loader'])}")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    model_type = config.get("model_type", "vae")
    print(f"   Model: {model_type.upper()}, params: {total_params:,}")

    # --- Train ---
    trainer = Trainer(model, config, device)
    results = trainer.train(data["train_loader"], data["val_loader"])

    print(f"\nTraining complete. Results in: {results['run_dir']}")


if __name__ == "__main__":
    main()
