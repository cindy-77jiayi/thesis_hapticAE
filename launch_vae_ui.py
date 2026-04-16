"""Launch the local 3-image semantic -> PC -> VAE reconstruction UI."""

from __future__ import annotations

import argparse

from src.llm import DEFAULT_OPENAI_MODEL
from src.ui import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the 3-image semantic to VAE reconstruction UI")
    parser.add_argument(
        "--frozen_manifest",
        type=str,
        required=True,
        help="Path to a frozen manifest JSON with config_path, checkpoint_path, and pca_dir.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--outputs_dir", type=str, default="outputs/vae_llm_ui")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(
        frozen_manifest=args.frozen_manifest,
        model=args.model,
        outputs_dir=args.outputs_dir,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        allowed_paths=[args.outputs_dir],
    )


if __name__ == "__main__":
    main()
