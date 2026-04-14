"""Launch the local 3-image OpenAI -> HapticGen validation UI."""

from __future__ import annotations

import argparse

from src.llm_hapticgen.ui import build_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the 3-image OpenAI to HapticGen validation UI")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--outputs_dir", type=str, default="outputs/hapticgen_llm_ui")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo(model=args.model, outputs_dir=args.outputs_dir)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
