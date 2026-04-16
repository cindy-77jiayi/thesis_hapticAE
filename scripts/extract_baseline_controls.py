"""Fit a post-hoc PCA baseline from pooled codec features."""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.data.loaders import build_dataloaders, build_model, load_checkpoint
from src.pipelines.control_baseline import (
    extract_pooled_features_and_sequences,
    fit_control_to_sequence_regressor,
    save_baseline_artifacts,
)
from src.pipelines.pca_control import fit_pca_pipeline, plot_sweep, sweep_axis
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Extract pooled codec controls and fit PCA baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)

    analysis_batch_size = int(config["data"].get("analysis_batch_size", 8))
    data = build_dataloaders(config, args.data_dir, batch_size=analysis_batch_size, full_dataset=True)
    print(f"Dataset size: {len(data['audio_files'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"✅ Loaded codec checkpoint: {args.checkpoint}")

    pooled_controls, seq_latents = extract_pooled_features_and_sequences(model, data["all_loader"], device)
    print(
        f"Extracted pooled controls: shape={pooled_controls.shape}, "
        f"mean={pooled_controls.mean():.4f}, std={pooled_controls.std():.4f}"
    )
    print(
        f"Extracted sequence latents: shape={seq_latents.shape}, "
        f"mean={seq_latents.mean():.4f}, std={seq_latents.std():.4f}"
    )

    np.save(os.path.join(args.output_dir, "pooled_controls.npy"), pooled_controls)
    pipe, Z_pca = fit_pca_pipeline(pooled_controls, n_components=args.n_components, save_dir=args.output_dir)

    regressor = fit_control_to_sequence_regressor(
        pooled_controls=pooled_controls,
        seq_latents=seq_latents,
        alpha=args.ridge_alpha,
    )
    artifacts = save_baseline_artifacts(args.output_dir, pooled_controls, seq_latents, regressor)
    print("Saved baseline artifacts:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")

    print(f"\n🏁 Baseline extraction complete. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
