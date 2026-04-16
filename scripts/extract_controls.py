"""Extract control latents from HapticCodec and fit PCA."""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.data.loaders import build_dataloaders, build_model, load_checkpoint
from src.pipelines.latent_extraction import extract_latent_vectors
from src.pipelines.pca_control import fit_pca_pipeline, sweep_axis, plot_sweep
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Extract control latents and fit PCA")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    analysis_batch_size = int(config["data"].get("analysis_batch_size", 8))
    data = build_dataloaders(config, args.data_dir, batch_size=analysis_batch_size, full_dataset=True)
    print(f"   Dataset size: {len(data['audio_files'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    Z = extract_latent_vectors(model, data["all_loader"], device)
    np.save(os.path.join(args.output_dir, "Z_ctrl.npy"), Z)
    pipe, _ = fit_pca_pipeline(Z, n_components=args.n_components, save_dir=args.output_dir)

    if args.sweep:
        for axis in range(args.n_components):
            result = sweep_axis(
                pipe,
                model,
                device,
                axis=axis,
                sweep_range=(-2.0, 2.0),
                n_steps=9,
                T=config["data"]["T"],
                sr=config["data"]["sr"],
            )
            plot_sweep(result, sr=config["data"]["sr"], save_path=os.path.join(args.output_dir, f"sweep_PC{axis+1}.png"))

    print(f"\n🏁 Control extraction complete. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
