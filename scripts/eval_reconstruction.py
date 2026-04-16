"""Evaluate HapticCodec reconstruction quality."""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate HapticCodec reconstruction quality")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    data = build_dataloaders(config, args.data_dir, batch_size=args.n_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    result = evaluate_reconstruction(
        model,
        data["val_loader"],
        device,
        n_samples=args.n_samples,
        model_type="codec",
        sr=config["data"]["sr"],
        clamp_range=config["loss"].get("clamp_range", 3.0),
    )
    print_metrics(result)
    std_ratios = [float(sample["std_ratio"]) for sample in result["per_sample"]]
    result["reconstruction_summary"]["mean_std_ratio"] = float(np.mean(std_ratios))
    print(f"Mean STD ratio: {result['reconstruction_summary']['mean_std_ratio']:.2%}")

    plot_waveform_comparison(
        result["x_np"],
        result["xhat_np"],
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

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(result["reconstruction_summary"], fp, indent=2)
    print(f"📁 Evaluation results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
