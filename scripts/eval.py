"""CLI entry point for evaluating a trained model."""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.data.loaders import build_dataloaders, build_model, load_checkpoint
from src.eval.evaluate import evaluate_reconstruction, print_metrics
from src.eval.visualize import plot_loss_curves, plot_waveform_comparison
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained haptic model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with key=value, e.g. data.train_split=0.9",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.set)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    split_manifest = os.path.join(checkpoint_dir, "data_split.json")
    split_manifest_path = split_manifest if os.path.exists(split_manifest) else None

    data = build_dataloaders(
        config,
        args.data_dir,
        batch_size=args.n_samples,
        split_manifest_path=split_manifest_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    is_vae = config.get("model_type", "vae") == "vae"
    print(f"Loaded checkpoint: {args.checkpoint}")

    result = evaluate_reconstruction(
        model,
        data["val_loader"],
        device,
        n_samples=args.n_samples,
        is_vae=is_vae,
        sr=config["data"]["sr"],
    )
    print_metrics(result)

    plot_waveform_comparison(
        result["x_np"],
        result["xhat_np"],
        save_path=os.path.join(args.output_dir, "waveforms.png"),
    )

    metrics_path = os.path.join(checkpoint_dir, "metrics.npz")
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path)
        plot_loss_curves(
            metrics["train_losses"].tolist(),
            metrics["val_losses"].tolist(),
            save_path=os.path.join(args.output_dir, "loss_curves.png"),
        )

    print(f"Evaluation results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
