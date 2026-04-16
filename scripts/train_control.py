"""Train the control branch on top of a frozen HapticCodec."""

import argparse
import json
import os

from _bootstrap import add_project_root

add_project_root()

import numpy as np
import torch
from tqdm import tqdm

from src.data.loaders import (
    build_dataloaders,
    build_model,
    load_checkpoint,
    load_codec_backbone_checkpoint,
)
from src.training.control_trainer import ControlTrainer, compute_metric_targets
from src.utils.config import load_config
from src.utils.seed import set_seed


def compute_metric_stats(loader, metric_names, sr: int) -> dict[str, list[float]]:
    rows = []
    for x in tqdm(loader, desc="Metric stats", leave=False):
        rows.append(compute_metric_targets(x, metric_names, sr).numpy())
    data = np.concatenate(rows, axis=0)
    return {
        "mean": data.mean(axis=0).tolist(),
        "std": np.maximum(data.std(axis=0), 1e-6).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train HapticCodec control stage")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--codec_checkpoint", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    config["output_dir"] = os.path.join(args.output_root, args.run_name)
    config["run_name"] = "control"

    set_seed(config.get("seed", 42))
    batch_size = int(config.get("control_training", {}).get("batch_size", config["training"]["batch_size"]))
    data = build_dataloaders(config, args.data_dir, batch_size=batch_size)
    metric_names = list(config["control_loss"]["metric_names"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)
    try:
        load_checkpoint(model, args.codec_checkpoint, device)
        print(f"✅ Loaded full codec checkpoint: {args.codec_checkpoint}")
    except RuntimeError as exc:
        if "size mismatch" not in str(exc):
            raise
        print("ℹ️ Control branch shape mismatch detected; loading codec backbone only.")
        load_codec_backbone_checkpoint(model, args.codec_checkpoint, device)
        print(f"✅ Loaded codec backbone checkpoint: {args.codec_checkpoint}")

    metric_stats = compute_metric_stats(data["train_loader"], metric_names, int(config["data"]["sr"]))
    stats_path = os.path.join(args.output_root, args.run_name, "control", "metric_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump({"metric_names": metric_names, **metric_stats}, fp, indent=2)
    print(f"📊 Saved metric stats: {stats_path}")

    trainer = ControlTrainer(model, config, device, metric_stats)
    results = trainer.train(data["train_loader"], data["val_loader"])
    print(f"\n🏁 Control training complete. Results in: {results['run_dir']}")


if __name__ == "__main__":
    main()
