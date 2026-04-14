"""Validation and payload helpers for segment-level LLM inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.semantic.pc_semantics import CANONICAL_SEMANTIC_ORDER


SEGMENT_DURATION_SECONDS = 0.5
ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_segment_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a segment manifest from disk."""
    manifest_path = Path(path).resolve()
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return validate_segment_manifest(payload, manifest_path=manifest_path)


def validate_segment_manifest(
    payload: dict[str, Any],
    *,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one segment manifest and return a normalized payload."""
    if not isinstance(payload, dict):
        raise ValueError("Segment manifest must be a JSON object.")

    manifest_dir = Path(manifest_path).resolve().parent if manifest_path else Path.cwd()
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Segment manifest must contain a 'segments' list.")
    if not (1 <= len(segments) <= 5):
        raise ValueError(f"Segment manifest must contain 1-5 segments, got {len(segments)}.")

    normalized_segments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict):
            raise ValueError(f"Segment #{index} must be an object.")

        segment_id = str(segment.get("segment_id", "")).strip()
        if not segment_id:
            raise ValueError(f"Segment #{index} is missing a non-empty 'segment_id'.")
        if segment_id in seen_ids:
            raise ValueError(f"Duplicate segment_id: {segment_id!r}")
        seen_ids.add(segment_id)

        start_s = _coerce_numeric(segment.get("start_s"), field=f"{segment_id}.start_s")
        end_s = _coerce_numeric(segment.get("end_s"), field=f"{segment_id}.end_s")
        duration = end_s - start_s
        if duration <= 0:
            raise ValueError(f"Segment {segment_id!r} must have end_s > start_s.")
        if abs(duration - SEGMENT_DURATION_SECONDS) > 1e-6:
            raise ValueError(
                f"Segment {segment_id!r} must be exactly {SEGMENT_DURATION_SECONDS:.1f}s, got {duration:.6f}s."
            )

        keyframes = segment.get("keyframes")
        if not isinstance(keyframes, list):
            raise ValueError(f"Segment {segment_id!r} must contain a 'keyframes' list.")
        if not (1 <= len(keyframes) <= 3):
            raise ValueError(f"Segment {segment_id!r} must contain 1-3 keyframes, got {len(keyframes)}.")

        normalized_keyframes: list[str] = []
        for frame_idx, raw_path in enumerate(keyframes, start=1):
            frame_path = _resolve_existing_path(raw_path, manifest_dir)
            if frame_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                raise ValueError(
                    f"Segment {segment_id!r} keyframe #{frame_idx} must be an image "
                    f"({sorted(ALLOWED_IMAGE_SUFFIXES)}), got {frame_path.suffix!r}."
                )
            normalized_keyframes.append(str(frame_path))

        normalized_segments.append(
            {
                "segment_id": segment_id,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "keyframes": normalized_keyframes,
                "notes": str(segment.get("notes", "") or ""),
                "event_name": str(segment.get("event_name", "") or ""),
                "context": str(segment.get("context", "") or ""),
            }
        )

    reference_video_path = payload.get("reference_video_path")
    normalized_reference: str | None = None
    if reference_video_path not in (None, ""):
        normalized_reference = str(_resolve_existing_path(reference_video_path, manifest_dir))

    target_duration_seconds = payload.get("target_duration_seconds")
    if target_duration_seconds in (None, ""):
        normalized_target_duration = float(len(normalized_segments) * SEGMENT_DURATION_SECONDS)
    else:
        normalized_target_duration = _coerce_numeric(target_duration_seconds, field="target_duration_seconds")
        expected_duration = float(len(normalized_segments) * SEGMENT_DURATION_SECONDS)
        if abs(normalized_target_duration - expected_duration) > 1e-6:
            raise ValueError(
                "target_duration_seconds must equal segment_count * 0.5s. "
                f"Expected {expected_duration:.1f}, got {normalized_target_duration:.6f}."
            )

    return {
        "reference_video_path": normalized_reference,
        "target_duration_seconds": float(normalized_target_duration),
        "segments": normalized_segments,
        "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
    }


def build_reconstruction_payload(
    segment_outputs: list[dict[str, Any]],
    *,
    run_name: str = "vae_balanced",
    target_sr: int = 8000,
    segment_duration_s: float = SEGMENT_DURATION_SECONDS,
) -> dict[str, Any]:
    """Build the Part 1 output payload consumed by later reconstruction stages."""
    if not segment_outputs:
        raise ValueError("segment_outputs must not be empty.")

    payload_segments: list[dict[str, Any]] = []
    for segment in segment_outputs:
        semantic_controls = segment["semantic_controls"]
        payload_segments.append(
            {
                "segment_id": str(segment["segment_id"]),
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
                "pc_vector": [float(v) for v in segment["pc_vector"]],
                "rationale": dict(segment.get("rationale", {})),
            }
        )

    return {
        "run_name": run_name,
        "target_sr": int(target_sr),
        "segment_duration_s": float(segment_duration_s),
        "segments": payload_segments,
    }


def _coerce_numeric(value: Any, *, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric, got {type(value).__name__}.")
    return float(value)


def _resolve_existing_path(raw_path: Any, base_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Expected a non-empty file path string.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path
