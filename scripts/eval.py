"""CLI entry point for evaluating a trained model.

Usage:
    python scripts/eval.py --config configs/vae_default.yaml --data_dir /path/to/wavs --checkpoint outputs/run/best_model.pt --output_dir outputs/eval
"""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.data.loaders import build_dataloaders, build_model, load_checkpoint
from src.eval.evaluate import evaluate_reconstruction, print_metrics
from src.eval.visualize import plot_loss_curves, plot_waveform_comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained haptic model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    data = build_dataloaders(
        config,
        args.data_dir,
        batch_size=args.n_samples,
        enable_train_augmentation=False,
    )

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    is_vae = config.get("model_type", "vae") == "vae"
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # --- Evaluate ---
    result = evaluate_reconstruction(
        model,
        data["val_loader"],
        device,
        n_samples=args.n_samples,
        is_vae=is_vae,
        sr=config["data"]["sr"],
    )
    print_metrics(result)

    # --- Plots ---
    plot_waveform_comparison(
        result["x_np"], result["xhat_np"],
        save_path=os.path.join(args.output_dir, "waveforms.png"),
    )

    metrics_path = os.path.join(os.path.dirname(args.checkpoint), "metrics.npz")
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path)
        plot_loss_curves(
            metrics["train_losses"].tolist(),
            metrics["val_losses"].tolist(),
            save_path=os.path.join(args.output_dir, "loss_curves.png"),
        )

    print(f"📁 Evaluation results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
