"""Generate one haptic signal from 8 PCA control values.

Usage:
    python scripts/generate_from_controls.py \
        --config configs/vae_balanced.yaml \
        --checkpoint outputs/vae_balanced/best_model.pt \
        --pca_dir outputs/pca \
        --controls_dir outputs/controls \
        --values 0.5,-0.2,0,0,0,0,0,0 \
        --output_dir outputs/generated
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf
import torch

from src.models.conv_vae import ConvVAE
from src.pipelines.pca_control import control_to_latent
from src.utils.config import load_config


def parse_values(values_str: str, expected_dim: int | None = None) -> np.ndarray:
    values = [float(v.strip()) for v in values_str.split(",") if v.strip() != ""]
    arr = np.array(values, dtype=np.float32)
    if expected_dim is not None and len(arr) != expected_dim:
        raise ValueError(f"Expected {expected_dim} control values, got {len(arr)}")
    return arr


def clamp_to_spec(values: np.ndarray, controls_spec: dict) -> tuple[np.ndarray, list[dict]]:
    clipped = values.copy()
    reports = []
    for i, ctrl in enumerate(controls_spec["controls"]):
        lo = ctrl["range"]["low"]
        hi = ctrl["range"]["high"]
        original = float(clipped[i])
        clipped[i] = np.clip(clipped[i], lo, hi)
        reports.append({
            "axis": i,
            "name": ctrl["name"],
            "semantic_tier": ctrl.get("semantic_tier", "Primary" if i < 4 else "Secondary"),
            "input": original,
            "used": float(clipped[i]),
            "range": [lo, hi],
            "clipped": abs(original - float(clipped[i])) > 1e-8,
        })
    return clipped, reports


def main():
    parser = argparse.ArgumentParser(description="Generate haptic signal from control knobs")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pca_dir", type=str, required=True, help="Directory containing pca_pipe.pkl")
    parser.add_argument("--controls_dir", type=str, required=True, help="Directory containing controls_spec.json")
    parser.add_argument("--values", type=str, required=True,
                        help="Comma-separated control values, e.g. '0.1,-0.2,0,0,0,0,0,0'")
    parser.add_argument("--output_dir", type=str, default="outputs/generated")
    parser.add_argument("--name", type=str, default="sample")
    parser.add_argument("--no_clamp", action="store_true", help="Disable range clamping by controls spec")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    with open(Path(args.pca_dir) / "pca_pipe.pkl", "rb") as f:
        pipe = pickle.load(f)

    spec_path = Path(args.controls_dir) / "controls_spec.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        controls_spec = json.load(f)

    n_controls = controls_spec["n_controls"]
    c = parse_values(args.values, expected_dim=n_controls)

    if not args.no_clamp:
        c_used, reports = clamp_to_spec(c, controls_spec)
    else:
        c_used, reports = c, []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    z_np = control_to_latent(pipe, c_used)
    z = torch.from_numpy(z_np).float().unsqueeze(0).to(device)

    with torch.no_grad():
        y = model.decode(z, target_len=data_cfg["T"]).squeeze().cpu().numpy()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_path = out_dir / f"{args.name}.wav"
    npy_path = out_dir / f"{args.name}.npy"
    meta_path = out_dir / f"{args.name}_meta.json"

    sf.write(wav_path, y, samplerate=data_cfg["sr"])
    np.save(npy_path, y)

    meta = {
        "controls_input": c.tolist(),
        "controls_used": c_used.tolist(),
        "reports": reports,
        "sr": data_cfg["sr"],
        "signal_length": len(y),
        "checkpoint": args.checkpoint,
        "config": args.config,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Generation complete")
    print(f"  Controls: {c_used.tolist()}")
    primary_used = [r["used"] for r in reports if r.get("semantic_tier") == "Primary"]
    secondary_used = [r["used"] for r in reports if r.get("semantic_tier") == "Secondary"]
    if reports:
        print(f"  Primary controls ({len(primary_used)}): {primary_used}")
        print(f"  Secondary controls ({len(secondary_used)}): {secondary_used}")
    print(f"  Saved WAV: {wav_path}")
    print(f"  Saved NPY: {npy_path}")
    print(f"  Saved META: {meta_path}")


if __name__ == "__main__":
    main()
