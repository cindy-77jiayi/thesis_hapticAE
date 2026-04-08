"""CLI entry point for training haptic signal models."""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import torch

from src.data.loaders import build_dataloaders, build_model
from src.training.checkpointing import prepare_run_dir
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train haptic signal model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None, help="Path to last_checkpoint.pt for resuming")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with key=value, e.g. training.epochs=20",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)
    config["output_dir"] = args.output_dir
    if args.resume:
        config["run_name"] = os.path.basename(os.path.dirname(os.path.abspath(args.resume)))

    set_seed(config.get("seed", 42))
    artifacts = prepare_run_dir(config)

    print(f"Collecting WAV files from: {args.data_dir}")
    data = build_dataloaders(
        config,
        args.data_dir,
        split_manifest_path=artifacts.split_manifest_path,
    )
    print(f"  Found {len(data['wav_files'])} clean WAV files")
    print(f"  Global RMS: {data['global_rms']:.6f}")
    print(f"  Batches: train={len(data['train_loader'])}, val={len(data['val_loader'])}")
    print(f"  Run dir: {artifacts.run_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    model_type = config.get("model_type", "vae")
    print(f"  Model: {model_type.upper()}, params: {total_params:,}")

    trainer = Trainer(model, config, device, artifacts=artifacts)
    if args.resume:
        trainer.restore(args.resume)
        print(f"  Resumed from: {args.resume}")

    results = trainer.train(data["train_loader"], data["val_loader"])
    print(f"\nTraining complete. Results in: {results['run_dir']}")


if __name__ == "__main__":
    main()
