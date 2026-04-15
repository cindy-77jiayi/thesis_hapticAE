"""Prepare WavCaps clips with a HapticGen-style preprocessing pipeline."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional UI dependency
    tqdm = None


MIN_ACCEPTABLE_AMPLITUDE = 0.05
MIN_ACCEPTABLE_995TH_AMPLITUDE = 0.02
WANTED_BIN_SIZE_SEC = 0.010
BASE_FREQ = 200.0
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")
RECORD_LIST_KEYS = ("data", "audios", "clips", "results", "dataset")
CAPTION_KEYS = (
    "chatgpt_caption",
    "caption",
    "description",
    "text",
    "raw_description",
    "title",
)
AUDIO_REF_KEYS = (
    "audio_path",
    "audio",
    "wav_path",
    "wav",
    "filename",
    "file_name",
    "id",
    "audio_id",
)


def prepare_wavcaps_haptic_dataset(
    metadata_dir: str,
    audio_dir: str,
    output_dir: str,
    output_sr: int = 8000,
    audio_extensions: tuple[str, ...] = DEFAULT_AUDIO_EXTENSIONS,
    create_variants: bool = False,
    n_variants: int = 4,
    openai_model: str = "gpt-4o-mini",
    limit: int | None = None,
    progress_every: int = 500,
) -> dict[str, int]:
    """Convert WavCaps audio clips into a filtered haptic-style dataset."""
    metadata_root = Path(metadata_dir)
    audio_root = Path(audio_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.jsonl"
    rejected_path = output_root / "rejected.jsonl"
    existing_manifest_paths = load_existing_manifest_paths(manifest_path)
    manifest_mode = "a" if existing_manifest_paths else "w"
    rejected_mode = "a" if rejected_path.exists() else "w"
    audio_index = build_audio_index(audio_root, audio_extensions)

    accepted = 0
    rejected = 0
    scanned = 0
    skipped_existing = 0
    total_records = count_wavcaps_records(metadata_root)
    if limit is not None:
        total_records = min(total_records, limit)

    if existing_manifest_paths:
        print(
            f"[prepare] resuming with {len(existing_manifest_paths)} existing manifest entries",
            flush=True,
        )

    progress = (
        tqdm(total=total_records, desc="prepare", unit="sample", dynamic_ncols=True)
        if tqdm is not None
        else None
    )

    with manifest_path.open(manifest_mode, encoding="utf-8") as manifest_fp, rejected_path.open(rejected_mode, encoding="utf-8") as rejected_fp:
        for record in iter_wavcaps_records(metadata_root):
            if limit is not None and scanned >= limit:
                break

            scanned += 1
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    accepted=accepted,
                    rejected=rejected,
                    skipped=skipped_existing,
                )
            if progress_every > 0 and scanned % progress_every == 0:
                print(
                    f"[prepare] scanned={scanned} accepted={accepted} rejected={rejected} skipped_existing={skipped_existing}",
                    flush=True,
                )
            caption = extract_caption(record["payload"])
            if not caption:
                rejected += 1
                rejected_fp.write(json.dumps({"reason": "missing_caption", **record}, ensure_ascii=False) + "\n")
                continue

            audio_path = resolve_audio_path(record["payload"], audio_index)
            if audio_path is None:
                rejected += 1
                rejected_fp.write(json.dumps({"reason": "missing_audio", **record}, ensure_ascii=False) + "\n")
                continue

            sample_id = slugify(f"{Path(audio_path).stem}_{record['record_index']:06d}")
            source_group = slugify(record["source_group"])
            sample_dir = output_root / source_group / sample_id
            wav_path = sample_dir / f"{sample_id}.wav"
            meta_path = sample_dir / f"{sample_id}.am1.json"
            if wav_path.exists() and meta_path.exists():
                skipped_existing += 1
                backfill_manifest_entry(
                    manifest_fp=manifest_fp,
                    existing_manifest_paths=existing_manifest_paths,
                    wav_path=wav_path,
                    meta_path=meta_path,
                )
                continue

            try:
                waveform, input_sr = load_audio_mono(audio_path)
                haptic_wave = audio_to_haptic(waveform, input_sr=input_sr, output_sr=output_sr)
            except Exception as exc:
                rejected += 1
                rejected_fp.write(
                    json.dumps(
                        {
                            "reason": "audio_processing_error",
                            "audio_path": audio_path,
                            "caption": caption,
                            "error": f"{type(exc).__name__}: {exc}",
                            **record,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                continue

            passed, metrics = passes_hapticgen_filter(haptic_wave)
            if not passed:
                rejected += 1
                rejected_fp.write(
                    json.dumps(
                        {
                            "reason": "below_amplitude_threshold",
                            "audio_path": audio_path,
                            "caption": caption,
                            "filter_metrics": metrics,
                            **record,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                continue

            prompt_variants = []
            if create_variants:
                prompt_variants = generate_prompt_variants(
                    caption,
                    n_variants=n_variants,
                    model=openai_model,
                )

            sample_dir.mkdir(parents=True, exist_ok=True)
            try:
                sf.write(wav_path, haptic_wave, output_sr, subtype="PCM_U8")
            except Exception as exc:
                rejected += 1
                rejected_fp.write(
                    json.dumps(
                        {
                            "reason": "write_error",
                            "audio_path": audio_path,
                            "caption": caption,
                            "error": f"{type(exc).__name__}: {exc}",
                            **record,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )
                continue

            metadata = {
                "filename": wav_path.name,
                "user_prompt": caption,
                "model": "Baseline-AudioGen",
                "vote": 1,
                "prompt_variant": prompt_variants[0] if prompt_variants else None,
                "prompt_variants": prompt_variants,
                "source_dataset": "WavCaps",
                "source_group": record["source_group"],
                "source_audio_path": audio_path,
                "source_metadata_file": record["metadata_file"],
                "filter_metrics": metrics,
            }
            with meta_path.open("w", encoding="utf-8") as fp:
                json.dump(metadata, fp, ensure_ascii=False, indent=2)

            manifest_fp.write(
                json.dumps(
                    {
                        "path": str(wav_path),
                        "metadata_path": str(meta_path),
                        **metadata,
                    },
                    ensure_ascii=False,
                ) + "\n"
            )
            existing_manifest_paths.add(str(meta_path))
            accepted += 1

    if progress is not None:
        progress.close()
    print(
        f"[prepare] complete scanned={scanned} accepted={accepted} rejected={rejected} skipped_existing={skipped_existing}",
        flush=True,
    )
    return {
        "scanned": scanned,
        "accepted": accepted,
        "rejected": rejected,
        "skipped_existing": skipped_existing,
    }


def iter_wavcaps_records(metadata_root: Path):
    """Yield normalized records from WavCaps metadata json files."""
    for json_path in sorted(metadata_root.rglob("*.json")):
        if json_path.parent.name.lower() == "blacklist":
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            records = extract_record_list(payload)
        except ValueError:
            continue
        source_group = json_path.parent.name
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            yield {
                "metadata_file": str(json_path),
                "source_group": source_group,
                "record_index": index,
                "payload": record,
            }


def count_wavcaps_records(metadata_root: Path) -> int:
    """Count records for progress reporting without holding them all in memory."""
    total = 0
    for json_path in sorted(metadata_root.rglob("*.json")):
        if json_path.parent.name.lower() == "blacklist":
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        try:
            records = extract_record_list(payload)
        except ValueError:
            continue
        total += sum(1 for record in records if isinstance(record, dict))
    return total


def extract_record_list(payload: Any) -> list[dict[str, Any]]:
    """Normalize a WavCaps metadata payload into a list of records."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in RECORD_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        dict_values = list(payload.values())
        if dict_values and all(isinstance(item, dict) for item in dict_values):
            return dict_values

        nested_records = []
        for value in payload.values():
            if isinstance(value, list):
                nested_records.extend(item for item in value if isinstance(item, dict))
        if nested_records:
            return nested_records

    raise ValueError("Unsupported WavCaps metadata structure")


def build_audio_index(audio_root: Path, extensions: tuple[str, ...]) -> dict[str, str]:
    """Build a lookup table for matching metadata records to audio files."""
    normalized_exts = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in extensions
    }
    index: dict[str, str] = {}
    for path in sorted(audio_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in normalized_exts:
            continue
        relative = path.relative_to(audio_root).as_posix().lower()
        basename = path.name.lower()
        stem = path.stem.lower()
        for key in {relative, basename, stem}:
            index.setdefault(key, str(path))
    return index


def load_existing_manifest_paths(manifest_path: Path) -> set[str]:
    """Collect metadata paths already listed in an existing manifest."""
    if not manifest_path.exists():
        return set()

    existing: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta_path = payload.get("metadata_path")
            if isinstance(meta_path, str) and meta_path:
                existing.add(meta_path)
    return existing


def backfill_manifest_entry(
    manifest_fp,
    existing_manifest_paths: set[str],
    wav_path: Path,
    meta_path: Path,
) -> None:
    """Re-add an existing sample to the manifest if it was produced before a crash."""
    meta_key = str(meta_path)
    if meta_key in existing_manifest_paths:
        return

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest_fp.write(
        json.dumps(
            {
                "path": str(wav_path),
                "metadata_path": meta_key,
                **metadata,
            },
            ensure_ascii=False,
        ) + "\n"
    )
    existing_manifest_paths.add(meta_key)


def extract_caption(record: dict[str, Any]) -> str | None:
    """Best-effort caption extraction from heterogeneous WavCaps records."""
    for key in CAPTION_KEYS:
        value = record.get(key)
        text = _coerce_text(value)
        if text:
            return text
    return None


def resolve_audio_path(record: dict[str, Any], audio_index: dict[str, str]) -> str | None:
    """Resolve an audio file path from a WavCaps metadata record."""
    for key in AUDIO_REF_KEYS:
        value = record.get(key)
        if not value:
            continue
        for candidate in candidate_audio_keys(str(value)):
            match = audio_index.get(candidate)
            if match:
                return match
    return None


def candidate_audio_keys(value: str) -> list[str]:
    """Generate lookup keys from a raw metadata audio reference."""
    raw = value.strip().replace("\\", "/").lower()
    basename = Path(raw).name.lower()
    stem = Path(basename).stem.lower()
    candidates = [raw, basename, stem]
    if "." not in basename:
        for ext in DEFAULT_AUDIO_EXTENSIONS:
            normalized_ext = ext if ext.startswith(".") else f".{ext}"
            candidates.append(f"{basename}{normalized_ext.lower()}")
            candidates.append(f"{stem}{normalized_ext.lower()}")
    return list(dict.fromkeys(candidates))


def audio_to_haptic(waveform: np.ndarray, input_sr: int, output_sr: int) -> np.ndarray:
    """Convert an audio waveform to a vibration signal using HapticGen's baseline."""
    wav_norm = np.asarray(waveform, dtype=np.float32).squeeze()
    if wav_norm.ndim != 1:
        wav_norm = librosa.to_mono(wav_norm)
    peak = float(np.max(np.abs(wav_norm))) if len(wav_norm) else 0.0
    if peak > 1e-8:
        wav_norm = wav_norm / peak
    return amp_env_on_wav_norm(wav_norm, input_sample_rate=input_sr, output_sample_rate=output_sr)


def load_audio_mono(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio with soundfile when possible, then fall back to librosa."""
    try:
        waveform, input_sr = sf.read(audio_path, always_2d=False)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform.squeeze(), int(input_sr)
    except Exception:
        waveform, input_sr = librosa.load(audio_path, sr=None, mono=True)
        return np.asarray(waveform, dtype=np.float32).squeeze(), int(input_sr)


def amp_env_on_wav_norm(
    wav_norm: np.ndarray,
    input_sample_rate: int,
    output_sample_rate: int,
) -> np.ndarray:
    """Amplitude-envelope baseline copied from HapticGen's audio-to-haptic path."""
    wav_norm = np.asarray(wav_norm, dtype=np.float32).squeeze()
    num_samples = len(wav_norm)
    if num_samples == 0:
        return np.zeros(1, dtype=np.float32)

    duration_sec = num_samples / input_sample_rate
    samples_per_bin = max(1, int(WANTED_BIN_SIZE_SEC * input_sample_rate))
    num_bins = max(1, num_samples // samples_per_bin)
    wav_chunks = np.array_split(wav_norm, num_bins)
    rms_bins = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in wav_chunks], dtype=np.float32)

    rms_max = float(np.max(rms_bins)) if len(rms_bins) else 0.0
    rms_norm = math.sqrt(2.0)
    rms_amplify = max(1.0, min(1.2, 1.0 / max(rms_max * rms_norm, 1e-8)))
    rms_norm_amp = rms_norm * rms_amplify
    out_samples = max(1, int(duration_sec * output_sample_rate))
    t = np.arange(out_samples, dtype=np.float32) / float(output_sample_rate)
    t_prog = np.minimum(1.0, t / max(duration_sec, 1e-8))
    bin_fi = t_prog * num_bins
    bin_lo = np.minimum(num_bins - 1, bin_fi.astype(np.int32))
    bin_hi = np.minimum(num_bins - 1, np.ceil(bin_fi).astype(np.int32))
    bin_fr = bin_fi - bin_lo

    rms = (
        (rms_bins[bin_lo] * (1.0 - bin_fr) + rms_bins[bin_hi] * bin_fr)
        * rms_norm_amp
    ).astype(np.float32)
    freq_offset = (rms - 0.3) * 100.0
    phase_delta = (2.0 * math.pi * (BASE_FREQ + freq_offset) / output_sample_rate).astype(np.float32)
    phase = np.mod(np.cumsum(phase_delta, dtype=np.float64), 2.0 * math.pi).astype(np.float32)
    return (rms * np.sin(phase)).astype(np.float32)


def passes_hapticgen_filter(signal: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Apply HapticGen's amplitude-based silent-sample filter."""
    signal = np.asarray(signal, dtype=np.float32)
    abs_signal = np.abs(signal)
    max_amp = float(np.max(abs_signal)) if len(abs_signal) else 0.0
    avg_amp = float(np.mean(abs_signal)) if len(abs_signal) else 0.0
    amp995 = float(np.quantile(abs_signal, 0.995)) if len(abs_signal) else 0.0
    passed = (
        max_amp >= MIN_ACCEPTABLE_AMPLITUDE
        and amp995 >= MIN_ACCEPTABLE_995TH_AMPLITUDE
    )
    return passed, {
        "max_amplitude": round(max_amp, 6),
        "avg_amplitude": round(avg_amp, 6),
        "amp_995": round(amp995, 6),
    }


def generate_prompt_variants(prompt: str, n_variants: int, model: str) -> list[str]:
    """Generate HapticGen-style prompt variants via OpenAI."""
    if n_variants <= 0:
        return []

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Prompt variant generation requires the `openai` package. "
            "Install dependencies or disable label augmentation."
        ) from exc

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.8,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate {n_variants} unique caption variants based on an input prompt for a "
                    "generative model. Use clear and natural 3rd person language. Avoid creative "
                    "flourishes and stick to straightforward captions. Avoid repetitive language and "
                    "focus on creating a variety that covers the spectrum of possible generations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = completion.choices[0].message.content or ""
    variants = []
    for line in content.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            variants.append(cleaned)
    return variants[:n_variants]


def slugify(value: str) -> str:
    """Convert arbitrary text into a filesystem-safe identifier."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return slug[:96] or "sample"


def _coerce_text(value: Any) -> str | None:
    """Normalize string-like values from metadata."""
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        for item in value:
            text = _coerce_text(item)
            if text:
                return text
    return None
