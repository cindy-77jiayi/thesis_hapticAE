"""Build control outputs for the pooled-feature PCA baseline."""

import argparse
import os
import pickle

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.data.loaders import build_model, load_checkpoint
from src.pipelines.control_baseline import PooledFeatureControlAdapter, load_regressor
from src.pipelines.control_spec import (
    build_controls_spec,
    build_controls_table_md,
    plot_metric_trends,
    plot_sweep_gallery,
    save_controls_spec,
)
from src.pipelines.pca_control import sweep_axis
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Build controls for the pooled-feature baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_sweep_steps", type=int, default=11)
    parser.add_argument("--max_primary_reuse", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)
    gallery_dir = os.path.join(args.output_dir, "pc_sweep_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = build_model(config, device)
    load_checkpoint(codec, args.checkpoint, device)
    regressor = load_regressor(os.path.join(args.baseline_dir, "control_to_seq.pkl"))
    model = PooledFeatureControlAdapter(codec, regressor, device)

    with open(os.path.join(args.baseline_dir, "controls_pca.pkl"), "rb") as fp:
        pipe = pickle.load(fp)
    Z_pca = np.load(os.path.join(args.baseline_dir, "Z_pca.npy"))
    evr = pipe.named_steps["pca"].explained_variance_ratio_

    spec = build_controls_spec(
        pipe,
        model,
        device,
        Z_pca,
        evr,
        T=config["data"]["T"],
        sr=config["data"]["sr"],
        n_sweep_steps=args.n_sweep_steps,
        max_primary_reuse=None if args.max_primary_reuse < 0 else args.max_primary_reuse,
    )
    save_controls_spec(spec, os.path.join(args.output_dir, "controls_spec.json"))

    sweep_results = []
    for ctrl in spec["controls"]:
        axis = ctrl["axis"]
        sweep_range = (ctrl["range"]["low"], ctrl["range"]["high"])
        sweep = sweep_axis(
            pipe,
            model,
            device,
            axis=axis,
            sweep_range=sweep_range,
            n_steps=args.n_sweep_steps,
            T=config["data"]["T"],
            sr=config["data"]["sr"],
            with_metrics=True,
        )
        sweep_results.append(sweep)

    plot_sweep_gallery(sweep_results, sr=config["data"]["sr"], save_dir=gallery_dir)
    plot_metric_trends(sweep_results, save_dir=args.output_dir)

    md = build_controls_table_md(spec)
    table_path = os.path.join(args.output_dir, "controls_table.md")
    with open(table_path, "w", encoding="utf-8") as fp:
        fp.write(md)
    print(f"Saved: {table_path}")
    print(f"\n🏁 Baseline controls saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
