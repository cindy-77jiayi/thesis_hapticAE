"""Build the frozen control specification: ranges, metrics, gallery, and table.

This script is the single entry point for solidifying the 8 PC controls.
It produces:
    - controls_spec.json   — machine-readable spec (ranges, metric profiles)
    - controls_table.md    — thesis-ready markdown table
    - pc_sweep_gallery/    — waveform + spectrogram PNGs for each PC
    - metric_trends.png    — how each metric varies across each PC

Usage:
    python scripts/build_controls.py \
        --config configs/vae_compact.yaml \
        --data_dir data/wavcaps_haptic_prepared \
        --checkpoint outputs/vae_compact/best_model.pt \
        --output_dir outputs/controls
"""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.loaders import build_model, load_checkpoint
from src.pipelines.latent_extraction import load_or_fit_pca
from src.pipelines.pca_control import sweep_axis
from src.pipelines.control_spec import (
    build_controls_spec,
    save_controls_spec,
    build_controls_table_md,
    plot_sweep_gallery,
    plot_metric_trends,
)


def main():
    parser = argparse.ArgumentParser(description="Build frozen control specification")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/controls")
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_sweep_steps", type=int, default=11,
                        help="Number of steps per sweep (default: 11)")
    parser.add_argument(
        "--max_primary_reuse",
        type=int,
        default=2,
        help="Maximum times the same metric can be used as primary across PCs "
             "(use negative value to disable cap).",
    )
    parser.add_argument("--pca_dir", type=str, default=None,
                        help="If provided, load existing PCA from this dir instead of re-fitting")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)
    gallery_dir = os.path.join(args.output_dir, "pc_sweep_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    data_cfg = config["data"]

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    load_checkpoint(model, args.checkpoint, device)
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # --- PCA ---
    pipe, Z_pca = load_or_fit_pca(
        model, config, args.data_dir, device,
        pca_dir=args.pca_dir,
        n_components=args.n_components,
        save_dir=args.output_dir,
    )

    evr = pipe.named_steps["pca"].explained_variance_ratio_

    # --- Build spec ---
    print("\n" + "=" * 60)
    print("Building control specification")
    print("=" * 60)
    spec = build_controls_spec(
        pipe, model, device, Z_pca, evr,
        T=data_cfg["T"], sr=data_cfg["sr"],
        n_sweep_steps=args.n_sweep_steps,
        max_primary_reuse=None if args.max_primary_reuse < 0 else args.max_primary_reuse,
    )
    save_controls_spec(spec, os.path.join(args.output_dir, "controls_spec.json"))

    # --- Sweep gallery ---
    print("\n" + "=" * 60)
    print("Generating sweep gallery (waveform + spectrogram)")
    print("=" * 60)
    sweep_results = []
    for ctrl in spec["controls"]:
        ax = ctrl["axis"]
        r = ctrl["range"]
        print(f"  PC{ax + 1}: sweeping [{r['low']:+.2f}, {r['high']:+.2f}]")
        sweep = sweep_axis(
            pipe, model, device,
            axis=ax, sweep_range=(r["low"], r["high"]),
            n_steps=args.n_sweep_steps,
            T=data_cfg["T"], sr=data_cfg["sr"],
            with_metrics=True,
        )
        sweep_results.append(sweep)

    plot_sweep_gallery(sweep_results, sr=data_cfg["sr"], save_dir=gallery_dir)
    plot_metric_trends(sweep_results, save_dir=args.output_dir)

    # --- Controls table (auto-labeled) ---
    print("\n" + "=" * 60)
    print("Generating controls_table.md")
    print("=" * 60)
    md = build_controls_table_md(spec)
    table_path = os.path.join(args.output_dir, "controls_table.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  Saved: {table_path}")

    print(f"\n🏁 All outputs saved to: {args.output_dir}")
    print(f"   controls_spec.json  — machine-readable spec")
    print(f"   controls_table.md   — thesis-ready table")
    print(f"   pc_sweep_gallery/   — waveform + spectrogram PNGs")
    print(f"   metric_trends.png   — metric trend plots")


if __name__ == "__main__":
    main()
