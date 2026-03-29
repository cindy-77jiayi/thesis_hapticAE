"""Import external waveform datasets into HapticGen-sidecar format.

This script converts arbitrary audio collections into:
  - 8kHz mono WAV chunks (default 10s, matching current in-repo data)
  - .am1.json metadata sidecars expected by src.data.preprocessing.collect_clean_wavs

Example:
    python scripts/import_external_wavs.py ^
      --src_dirs D:/fsd50k/dev_audio D:/fsd50k/eval_audio ^
      --dst_dir hapticgen-dataset/external_fsd50k_ccby ^
      --model_tag External-FSD50K-CCBY ^
      --source_name FSD50K
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aif", ".aiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert external audio files into 8k mono chunks with .am1.json sidecars.",
    )
    parser.add_argument(
        "--src_dirs",
        nargs="+",
        required=True,
        help="One or more source directories to scan recursively for audio files.",
    )
    parser.add_argument(
        "--dst_dir",
        required=True,
        help="Output directory where chunked WAV + .am1.json files will be written.",
    )
    parser.add_argument(
        "--source_name",
        default="ExternalDataset",
        help="Dataset name to store in metadata (for traceability).",
    )
    parser.add_argument(
        "--model_tag",
        default="ExternalWaveform",
        help="Metadata model field used by training filters (accepted_models).",
    )
    parser.add_argument("--target_sr", type=int, default=8000, help="Target sample rate.")
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=10.0,
        help="Chunk length in seconds. Default 10s to match current HapticGen files.",
    )
    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=10.0,
        help="Hop length in seconds between chunks.",
    )
    parser.add_argument(
        "--min_input_seconds",
        type=float,
        default=1.0,
        help="Skip input files shorter than this duration before chunking.",
    )
    parser.add_argument(
        "--min_chunk_rms",
        type=float,
        default=1e-4,
        help="Drop low-energy chunks below this RMS.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional cap on number of source files (0 means no cap).",
    )
    parser.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=0,
        help="Optional cap on chunks produced per source file (0 means no cap).",
    )
    parser.add_argument(
        "--include_tail_chunk",
        action="store_true",
        help="Also include a tail-aligned chunk when hop does not land on the end.",
    )
    return parser.parse_args()


def iter_audio_files(src_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for src in src_dirs:
        root = Path(src)
        if not root.exists():
            print(f"[WARN] Source dir does not exist: {root}")
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                files.append(p)
    files.sort()
    return files


def chunk_audio(
    y: np.ndarray,
    chunk_len: int,
    hop_len: int,
    include_tail_chunk: bool,
) -> list[np.ndarray]:
    n = int(y.shape[0])
    if n <= chunk_len:
        out = np.zeros(chunk_len, dtype=np.float32)
        out[:n] = y[:n]
        return [out]

    starts = list(range(0, n - chunk_len + 1, hop_len))
    if include_tail_chunk:
        tail_start = n - chunk_len
        if starts[-1] != tail_start:
            starts.append(tail_start)

    chunks = [y[s : s + chunk_len].astype(np.float32) for s in starts]
    return chunks


def stable_chunk_name(source_file: Path, chunk_idx: int) -> str:
    key = f"{source_file.as_posix()}::{chunk_idx}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:16]
    stem = source_file.stem.replace(" ", "_")
    return f"{stem}_{digest}"


def main() -> None:
    args = parse_args()
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    chunk_len = int(round(args.target_sr * args.chunk_seconds))
    hop_len = int(round(args.target_sr * args.hop_seconds))
    min_input_len = int(round(args.target_sr * args.min_input_seconds))

    if chunk_len <= 0 or hop_len <= 0:
        raise ValueError("chunk_seconds and hop_seconds must produce positive sample counts.")

    source_files = iter_audio_files(args.src_dirs)
    if args.max_files > 0:
        source_files = source_files[: args.max_files]

    print(f"Found {len(source_files)} source files")
    print(f"Output dir: {dst_dir}")
    print(
        f"Chunking: {args.chunk_seconds:.3f}s window, {args.hop_seconds:.3f}s hop, "
        f"target_sr={args.target_sr}"
    )

    total_chunks = 0
    kept_files = 0
    skipped_short = 0
    skipped_silent = 0
    failed = 0

    for i, src in enumerate(source_files, start=1):
        try:
            y, sr = librosa.load(str(src), sr=None, mono=True)
            y = y.astype(np.float32)
            if sr != args.target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.target_sr).astype(np.float32)
                sr = args.target_sr
            if y.shape[0] < min_input_len:
                skipped_short += 1
                continue
            if not np.isfinite(y).all():
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            chunks = chunk_audio(y, chunk_len, hop_len, args.include_tail_chunk)
            if args.max_chunks_per_file > 0:
                chunks = chunks[: args.max_chunks_per_file]

            kept_here = 0
            for cidx, chunk in enumerate(chunks):
                chunk = chunk - float(np.mean(chunk))
                rms = float(np.sqrt(np.mean(chunk**2) + 1e-12))
                if rms < args.min_chunk_rms:
                    skipped_silent += 1
                    continue

                stem = stable_chunk_name(src, cidx)
                shard = stem[:2]
                out_subdir = dst_dir / shard
                out_subdir.mkdir(parents=True, exist_ok=True)

                wav_name = f"{stem}.wav"
                wav_path = out_subdir / wav_name
                meta_path = out_subdir / f"{stem}.am1.json"

                sf.write(str(wav_path), chunk, samplerate=sr, subtype="PCM_16")

                metadata = {
                    "filename": wav_name,
                    "user_prompt": f"external import from {args.source_name}",
                    "model": args.model_tag,
                    "vote": 1,
                    "source_dataset": args.source_name,
                    "source_path": str(src),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                total_chunks += 1
                kept_here += 1

            if kept_here > 0:
                kept_files += 1

            if i % 200 == 0:
                print(f"[{i}/{len(source_files)}] chunks={total_chunks}, kept_files={kept_files}")
        except Exception:
            failed += 1

    print("Done")
    print(f"  Source files scanned: {len(source_files)}")
    print(f"  Source files with >=1 kept chunk: {kept_files}")
    print(f"  Kept chunks: {total_chunks}")
    print(f"  Skipped short files: {skipped_short}")
    print(f"  Skipped silent chunks: {skipped_silent}")
    print(f"  Failed files: {failed}")
    print(f"  model_tag to use in config.accepted_models: {args.model_tag}")


if __name__ == "__main__":
    main()

