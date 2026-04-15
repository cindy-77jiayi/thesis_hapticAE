"""Prepare a HapticGen-style haptic dataset from WavCaps audio and metadata.

Usage:
    python scripts/prepare_wavcaps_haptic_dataset.py \
        --metadata_dir /path/to/WavCaps/data/json_files \
        --audio_dir /path/to/WavCaps/audio \
        --output_dir data/wavcaps_haptic_prepared \
        --create_variants
"""

import argparse

from _bootstrap import add_project_root

add_project_root()

from src.data.wavcaps_hapticgen import prepare_wavcaps_haptic_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare WavCaps with HapticGen-style filtering, label augmentation, and audio-to-haptic conversion",
    )
    parser.add_argument("--metadata_dir", type=str, required=True, help="WavCaps metadata root, e.g. data/json_files")
    parser.add_argument("--audio_dir", type=str, required=True, help="Root directory containing WavCaps audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for converted haptic wavs and metadata")
    parser.add_argument("--output_sr", type=int, default=8000, help="Target sample rate for converted haptic waveforms")
    parser.add_argument(
        "--audio_extensions",
        nargs="*",
        default=[".wav", ".flac", ".mp3", ".ogg", ".m4a"],
        help="Audio extensions to index under --audio_dir",
    )
    parser.add_argument(
        "--create_variants",
        action="store_true",
        help="Generate HapticGen-style prompt variants with OpenAI and store them in metadata",
    )
    parser.add_argument("--n_variants", type=int, default=4, help="Number of prompt variants to request")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model name for prompt augmentation")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed metadata records")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=500,
        help="Print progress every N scanned records; set 0 to disable",
    )
    args = parser.parse_args()

    summary = prepare_wavcaps_haptic_dataset(
        metadata_dir=args.metadata_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        output_sr=args.output_sr,
        audio_extensions=tuple(args.audio_extensions),
        create_variants=args.create_variants,
        n_variants=args.n_variants,
        openai_model=args.openai_model,
        limit=args.limit,
        progress_every=args.progress_every,
    )

    print("Preparation complete")
    print(f"  scanned:  {summary['scanned']}")
    print(f"  accepted: {summary['accepted']}")
    print(f"  rejected: {summary['rejected']}")
    print(f"  skipped:  {summary.get('skipped_existing', 0)}")
    print(f"  output:   {args.output_dir}")


if __name__ == "__main__":
    main()
