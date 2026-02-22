"""CLI entry point for evaluating a trained model.

Usage:
    python scripts/eval.py --config configs/vae_default.yaml --data_dir /path/to/wavs --checkpoint outputs/run/best_model.pt --output_dir outputs/eval
"""

import argparse
import os
import sys

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
    data_cfg = config["data"]
    wav_files = collect_clean_wavs(args.data_dir)
    assert len(wav_files) > 0

    N = len(wav_files)
    perm = np.random.permutation(N)
    split = int(data_cfg["train_split"] * N)
    train_files = [wav_files[i] for i in perm[:split]]
    val_files = [wav_files[i] for i in perm[split:]]

    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

    val_ds = HapticWavDataset(
        val_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
    )
    val_loader = DataLoader(val_ds, batch_size=args.n_samples, shuffle=False)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model_type = config.get("model_type", "vae")
    is_vae = model_type == "vae"

    if is_vae:
        model = ConvVAE(
            T=data_cfg["T"],
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
            first_kernel=model_cfg.get("first_kernel", 25),
            kernel_size=model_cfg.get("kernel_size", 9),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
        )
    else:
        model = ConvAE(
            T=data_cfg["T"],
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
        )

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # --- Evaluate ---
    result = evaluate_reconstruction(model, val_loader, device, n_samples=args.n_samples, is_vae=is_vae)
    print_metrics(result)

    # --- Plots ---
    plot_waveform_comparison(
        result["x_np"], result["xhat_np"],
        save_path=os.path.join(args.output_dir, "waveforms.png"),
    )

    # Load metrics if available
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
