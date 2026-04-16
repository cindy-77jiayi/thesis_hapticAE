"""Train the HapticCodec reconstruction branch."""

import argparse
import os

from _bootstrap import add_project_root

add_project_root()

import torch

from src.data.loaders import build_dataloaders, build_model
from src.training.codec_trainer import CodecTrainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Train HapticCodec reconstruction stage")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config["model_type"] = "codec"
    config["output_dir"] = os.path.join(args.output_root, args.run_name)
    config["run_name"] = "codec"

    set_seed(config.get("seed", 42))
    print(f"📂 Collecting audio files from: {args.data_dir}")
    data = build_dataloaders(config, args.data_dir)
    print(f"   Found {len(data['audio_files'])} audio files")
    print(f"   Global RMS: {data['global_rms']:.6f}")
    print(f"   Batches: train={len(data['train_loader'])}, val={len(data['val_loader'])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")
    model = build_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model: HapticCodec, params: {total_params:,}")

    trainer = CodecTrainer(model, config, device)
    results = trainer.train(data["train_loader"], data["val_loader"])
    print(f"\n🏁 Codec training complete. Results in: {results['run_dir']}")


if __name__ == "__main__":
    main()
