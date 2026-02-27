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
        --data_dir /path/to/wavs \
        --checkpoint outputs/vae_compact/best_model.pt \
        --output_dir outputs/controls
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.preprocessing import collect_clean_wavs, estimate_global_rms
from src.data.dataset import HapticWavDataset
from src.models.conv_vae import ConvVAE
from src.pipelines.latent_extraction import extract_latent_vectors
from src.pipelines.pca_control import fit_pca_pipeline
from src.pipelines.control_spec import (
    build_controls_spec,
    save_controls_spec,
    build_controls_table_md,
    sweep_with_metrics,
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
    parser.add_argument("--pca_dir", type=str, default=None,
                        help="If provided, load existing PCA from this dir instead of re-fitting")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    os.makedirs(args.output_dir, exist_ok=True)
    gallery_dir = os.path.join(args.output_dir, "pc_sweep_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    # --- Data ---
    data_cfg = config["data"]
    wav_files = collect_clean_wavs(args.data_dir)
    assert len(wav_files) > 0, f"No WAV files found in {args.data_dir}"

    N = len(wav_files)
    perm = np.random.permutation(N)
    train_files = [wav_files[i] for i in perm[:int(data_cfg["train_split"] * N)]]
    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

    all_ds = HapticWavDataset(
        wav_files, T=data_cfg["T"], sr_expect=data_cfg["sr"],
        global_rms=global_rms, scale=data_cfg["scale"],
    )
    all_loader = DataLoader(all_ds, batch_size=64, shuffle=False, drop_last=False)

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config["model"]
    model = ConvVAE(
        T=data_cfg["T"],
        latent_dim=model_cfg["latent_dim"],
        channels=tuple(model_cfg["channels"]),
        first_kernel=model_cfg.get("first_kernel", 25),
        kernel_size=model_cfg.get("kernel_size", 9),
        activation=model_cfg.get("activation", "leaky_relu"),
        norm=model_cfg.get("norm", "group"),
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")
    print(f"   Latent dim: {model.latent_dim}, Dataset size: {len(all_ds)}")

    # --- PCA ---
    import pickle
    if args.pca_dir and os.path.exists(os.path.join(args.pca_dir, "pca_pipe.pkl")):
        print(f"\n📦 Loading existing PCA from {args.pca_dir}")
        with open(os.path.join(args.pca_dir, "pca_pipe.pkl"), "rb") as f:
            pipe = pickle.load(f)
        Z_pca = np.load(os.path.join(args.pca_dir, "Z_pca.npy"))
        Z = np.load(os.path.join(args.pca_dir, "Z.npy"))
    else:
        print("\n" + "=" * 60)
        print("Extracting latent vectors")
        print("=" * 60)
        Z = extract_latent_vectors(model, all_loader, device)
        np.save(os.path.join(args.output_dir, "Z.npy"), Z)

        print("\n" + "=" * 60)
        print(f"Fitting PCA ({Z.shape[1]}D → {args.n_components}D)")
        print("=" * 60)
        pipe, Z_pca = fit_pca_pipeline(Z, n_components=args.n_components, save_dir=args.output_dir)

    evr = pipe.named_steps["pca"].explained_variance_ratio_

    # --- Build spec ---
    print("\n" + "=" * 60)
    print("Building control specification")
    print("=" * 60)
    spec = build_controls_spec(
        pipe, model, device, Z_pca, evr,
        T=data_cfg["T"], sr=data_cfg["sr"],
        n_sweep_steps=args.n_sweep_steps,
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
        sweep = sweep_with_metrics(
            pipe, model, device,
            axis=ax, sweep_range=(r["low"], r["high"]),
            n_steps=args.n_sweep_steps,
            T=data_cfg["T"], sr=data_cfg["sr"],
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
