"""CLI: segment manifest -> OpenAI semantics -> PC vectors -> local VAE reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.llm import DEFAULT_OPENAI_MODEL
from src.pipelines.semantic_reconstruction import run_manifest_reconstruction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run segment manifest -> OpenAI semantic controls -> PC vectors -> local VAE reconstruction"
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to one segment manifest JSON file")
    parser.add_argument(
        "--frozen_manifest",
        type=str,
        required=True,
        help="Path to a frozen manifest JSON with config_path, checkpoint_path, and pca_dir.",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/llm_to_haptic")
    parser.add_argument("--prompt_template", type=str, default="llm/prompt_template.md")
    parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_manifest_reconstruction(
        manifest_path=args.manifest,
        run_dir=output_dir,
        frozen_manifest_path=args.frozen_manifest,
        prompt_template=args.prompt_template,
        model=args.model,
    )
    print(f"Saved: {output_dir / 'normalized_segment_manifest.json'}")
    print(f"Saved: {output_dir / 'reconstruction_payload.json'}")
    print(f"Saved: {result['generated_wav']}")
    print(f"Saved: {result['waveform_image']}")
    print("Semantic reconstruction pipeline finished.")


if __name__ == "__main__":
    main()
