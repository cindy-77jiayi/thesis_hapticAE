"""Import FSD50K from Hugging Face into HapticGen-compatible sidecar format.

This script writes:
  - WAV files (mono, target sample rate)
  - .am1.json sidecars expected by src.data.preprocessing.collect_clean_wavs

Example:
  python scripts/import_hf_fsd50k.py \
    --repo_id Fhrozen/FSD50k \
    --split train \
    --num_samples 6000 \
    --dst_dir /content/hapticgen-dataset/external_fsd50k
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset
from huggingface_hub import HfFileSystem


_HF_FS = HfFileSystem()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import FSD50K clips from Hugging Face into .wav + .am1.json format.",
    )
    parser.add_argument("--repo_id", type=str, default="Fhrozen/FSD50k")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dst_dir", type=str, required=True)
    parser.add_argument("--audio_column", type=str, default="audio")
    parser.add_argument("--num_samples", type=int, default=4000)
    parser.add_argument("--target_sr", type=int, default=8000)
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use streaming mode (default: True).",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument("--max_seconds", type=float, default=10.0)
    parser.add_argument("--min_rms", type=float, default=1e-4)
    parser.add_argument("--model_tag", type=str, default="ExternalFSD50K")
    parser.add_argument("--source_name", type=str, default="FSD50K")
    return parser.parse_args()


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # Handle both (channels, time) and (time, channels).
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            return np.mean(y, axis=0)
        return np.mean(y, axis=1)
    return y.reshape(-1)


def _decode_audio_bytes(raw_bytes: bytes) -> tuple[np.ndarray, int] | None:
    try:
        y, sr2 = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        y = _to_mono(y)
        if y.size == 0:
            return None
        return y, int(sr2)
    except Exception:
        try:
            y, sr2 = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0:
                return None
            return y, int(sr2)
        except Exception:
            return None


def _read_hf_uri_bytes(hf_uri: str) -> bytes | None:
    if not hf_uri.startswith("hf://"):
        return None
    try:
        hf_path = hf_uri[len("hf://") :]
        with _HF_FS.open(hf_path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _load_audio_path(raw_path: str) -> tuple[np.ndarray, int] | None:
    if raw_path.startswith("hf://"):
        raw_bytes = _read_hf_uri_bytes(raw_path)
        if raw_bytes is None:
            return None
        return _decode_audio_bytes(raw_bytes)
    try:
        y, sr2 = sf.read(raw_path, dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        y = _to_mono(y)
        if y.size == 0:
            return None
        return y, int(sr2)
    except Exception:
        try:
            y, sr2 = librosa.load(raw_path, sr=None, mono=True)
            y = np.asarray(y, dtype=np.float32)
            if y.size == 0:
                return None
            return y, int(sr2)
        except Exception:
            return None


def _extract_audio(row: dict[str, Any], audio_column: str) -> tuple[np.ndarray, int] | None:
    audio_obj = row.get(audio_column)
    if isinstance(audio_obj, str):
        return _load_audio_path(audio_obj)
    if not isinstance(audio_obj, dict):
        return None

    arr = audio_obj.get("array")
    sr = audio_obj.get("sampling_rate")
    if arr is None or sr is None:
        raw_bytes = audio_obj.get("bytes")
        raw_path = audio_obj.get("path")
        if raw_bytes is not None:
            return _decode_audio_bytes(raw_bytes)
        if raw_path:
            return _load_audio_path(raw_path)
        return None

    y = np.asarray(arr, dtype=np.float32)
    y = _to_mono(y)
    if y.size == 0:
        return None
    return y, int(sr)


def _iter_examples(ds: Any, num_samples: int) -> Iterable[dict[str, Any]]:
    if isinstance(ds, IterableDataset):
        # In some datasets versions/environments, iterating an IterableDataset with
        # Audio features can trigger torch/torchcodec-dependent feature encoding.
        # Reading raw rows from _ex_iterable avoids that path.
        ex_iterable = getattr(ds, "_ex_iterable", None)
        if ex_iterable is not None:
            for i, item in enumerate(ex_iterable):
                if i >= num_samples:
                    break
                if isinstance(item, tuple) and len(item) == 2:
                    _, row = item
                else:
                    row = item
                if isinstance(row, dict):
                    yield row
            return
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            yield row
    else:
        n = min(num_samples, len(ds))
        for i in range(n):
            yield ds[i]


def main() -> None:
    args = parse_args()
    if args.no_streaming:
        args.streaming = False

    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {args.repo_id} split={args.split} "
        f"(streaming={args.streaming}, num_samples={args.num_samples})"
    )
    ds_dict = load_dataset(args.repo_id, streaming=args.streaming)
    available_splits = list(ds_dict.keys())
    chosen_split = args.split
    if chosen_split not in ds_dict:
        if "validation" in ds_dict:
            chosen_split = "validation"
        elif len(available_splits) > 0:
            chosen_split = available_splits[0]
        else:
            raise ValueError(f"No splits found in dataset: {args.repo_id}")
        print(
            f"[WARN] Requested split '{args.split}' not found. "
            f"Using '{chosen_split}' instead. Available: {available_splits}"
        )
    ds = ds_dict[chosen_split]
    features = getattr(ds, "features", None)
    if features is not None and args.audio_column not in features:
        available = ", ".join(features.keys())
        raise KeyError(
            f"Audio column '{args.audio_column}' not found. Available columns: {available}"
        )
    try:
        ds = ds.cast_column(args.audio_column, Audio(decode=False))
    except Exception as e:
        print(f"[WARN] cast_column failed ({type(e).__name__}): {e}")
        print("[WARN] Continuing without explicit cast; expecting audio rows with path/bytes.")

    max_len = int(round(args.max_seconds * args.target_sr))

    kept = 0
    skipped_bad = 0
    skipped_silent = 0

    for i, row in enumerate(_iter_examples(ds, args.num_samples)):
        extracted = _extract_audio(row, args.audio_column)
        if extracted is None:
            skipped_bad += 1
            continue
        y, sr = extracted

        if sr != args.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.target_sr).astype(np.float32)
            sr = args.target_sr

        y = y - float(np.mean(y))
        if y.shape[0] > max_len:
            y = y[:max_len]

        rms = float(np.sqrt(np.mean(y**2) + 1e-12))
        if not np.isfinite(rms) or rms < args.min_rms:
            skipped_silent += 1
            continue

        stem = f"fsd50k_{i:07d}"
        shard = f"{(i // 1000):03d}"
        out_subdir = dst / shard
        out_subdir.mkdir(parents=True, exist_ok=True)

        wav_name = f"{stem}.wav"
        wav_path = out_subdir / wav_name
        meta_path = out_subdir / f"{stem}.am1.json"

        sf.write(str(wav_path), y.astype(np.float32), samplerate=sr, subtype="PCM_16")

        meta = {
            "filename": wav_name,
            "user_prompt": f"external import from {args.source_name}",
            "model": args.model_tag,
            "vote": 1,
            "source_dataset": args.source_name,
            "source_repo": args.repo_id,
            "source_split": chosen_split,
            "source_index": i,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        kept += 1
        if kept % 500 == 0:
            print(f"  kept={kept}, skipped_bad={skipped_bad}, skipped_silent={skipped_silent}")

    print("Done")
    print(f"  output_dir: {dst}")
    print(f"  kept: {kept}")
    print(f"  skipped_bad: {skipped_bad}")
    print(f"  skipped_silent: {skipped_silent}")
    print(f"  model_tag (config accepted_models): {args.model_tag}")


if __name__ == "__main__":
    main()
