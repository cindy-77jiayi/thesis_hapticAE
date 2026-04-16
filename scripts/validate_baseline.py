"""Run extended validation for the pooled-feature PCA baseline."""

import argparse
import json
import os
import pickle

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch

from src.data.loaders import build_model, load_checkpoint
from src.eval.pc_validation import (
    compute_cross_influence,
    compute_effect_sizes,
    plot_selectivity_bar,
    print_cross_influence_report,
    print_effect_size_report,
)
from src.pipelines.control_baseline import PooledFeatureControlAdapter, load_regressor
from src.pipelines.control_spec import compute_control_ranges
from src.pipelines.pca_control import sweep_axis
from src.utils.config import load_config
from src.utils.seed import set_seed

from scripts.validate_extended import (
    compare_references,
    compute_extended_bindings,
    generate_table_v2,
    plot_extended_heatmap,
)


def main():
    parser = argparse.ArgumentParser(description="Extended validation for pooled-feature PCA baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--controls_dir", type=str, required=True)
    parser.add_argument("--n_components", type=int, default=8)
    parser.add_argument("--n_sweep_steps", type=int, default=21)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.controls_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = build_model(config, device)
    load_checkpoint(codec, args.checkpoint, device)
    regressor = load_regressor(os.path.join(args.baseline_dir, "control_to_seq.pkl"))
    model = PooledFeatureControlAdapter(codec, regressor, device)

    with open(os.path.join(args.baseline_dir, "controls_pca.pkl"), "rb") as fp:
        pipe = pickle.load(fp)
    Z_pca = np.load(os.path.join(args.baseline_dir, "Z_pca.npy"))
    evr = pipe.named_steps["pca"].explained_variance_ratio_
    ranges = compute_control_ranges(Z_pca)

    sweep_origin = []
    for axis in range(args.n_components):
        r = ranges[axis]
        sweep_origin.append(
            sweep_axis(
                pipe,
                model,
                device,
                axis=axis,
                sweep_range=(r["p5"], r["p95"]),
                n_steps=args.n_sweep_steps,
                T=config["data"]["T"],
                sr=config["data"]["sr"],
                with_metrics=True,
            )
        )

    bindings = compute_extended_bindings(sweep_origin)
    plot_extended_heatmap(
        bindings,
        save_path=os.path.join(args.output_dir, "monotonicity_extended_heatmap.png"),
    )

    z_pca_mean = Z_pca.mean(axis=0).astype(np.float32)
    metric_names = list(sweep_origin[0]["metrics"][0].keys())
    sweep_mean = []
    for axis in range(args.n_components):
        r = ranges[axis]
        sweep_mean.append(
            sweep_axis(
                pipe,
                model,
                device,
                axis=axis,
                sweep_range=(r["p5"], r["p95"]),
                n_steps=args.n_sweep_steps,
                T=config["data"]["T"],
                sr=config["data"]["sr"],
                reference=z_pca_mean,
                with_metrics=True,
            )
        )
    ref_comparison = compare_references(sweep_origin, sweep_mean, metric_names)

    auto_bindings = {}
    for pc in bindings["pc_names"]:
        top = [m for m in bindings["ranked_bindings"][pc] if m["significant"] and m["metric"] != "peak_amplitude"][:2]
        auto_bindings[pc] = [m["metric"] for m in top] if top else ["rms_energy"]
    cross = compute_cross_influence(bindings, auto_bindings)
    print_cross_influence_report(cross)
    plot_selectivity_bar(cross, save_path=os.path.join(args.output_dir, "selectivity_extended.png"))

    effects = compute_effect_sizes(sweep_origin)
    print_effect_size_report(effects)

    binding_json = {
        "n_metrics": len(bindings["metric_names"]),
        "metric_names": bindings["metric_names"],
        "ranked_bindings": bindings["ranked_bindings"],
        "reference_comparison": ref_comparison,
        "cross_influence": cross,
        "effect_sizes": effects,
    }
    with open(os.path.join(args.output_dir, "extended_metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(binding_json, fp, indent=2, default=str)

    generate_table_v2(
        bindings,
        alignment=None,
        evr=evr,
        ref_comparison=ref_comparison,
        save_path=os.path.join(args.controls_dir, "controls_table_v2.md"),
    )
    print(f"\n🏁 Baseline validation complete. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
