from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.llm.manifest import SEGMENT_DURATION_SECONDS, load_segment_manifest


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "segment_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_load_segment_manifest_accepts_one_segment(tmp_path: Path):
    frame_path = tmp_path / "frame_01.png"
    _touch_image(frame_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "target_duration_seconds": 0.5,
            "segments": [
                {
                    "segment_id": "seg_01",
                    "start_s": 0.0,
                    "end_s": 0.5,
                    "keyframes": [str(frame_path)],
                }
            ],
        },
    )

    manifest = load_segment_manifest(manifest_path)

    assert manifest["target_duration_seconds"] == SEGMENT_DURATION_SECONDS
    assert len(manifest["segments"]) == 1
    assert manifest["segments"][0]["keyframes"] == [str(frame_path.resolve())]


def test_load_segment_manifest_rejects_zero_segments(tmp_path: Path):
    manifest_path = _write_manifest(tmp_path, {"segments": []})
    with pytest.raises(ValueError, match="1-5 segments"):
        load_segment_manifest(manifest_path)


def test_load_segment_manifest_rejects_more_than_five_segments(tmp_path: Path):
    frame_path = tmp_path / "frame_01.png"
    _touch_image(frame_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "target_duration_seconds": 3.0,
            "segments": [
                {
                    "segment_id": f"seg_{idx}",
                    "start_s": idx * 0.5,
                    "end_s": (idx + 1) * 0.5,
                    "keyframes": [str(frame_path)],
                }
                for idx in range(6)
            ],
        },
    )
    with pytest.raises(ValueError, match="1-5 segments"):
        load_segment_manifest(manifest_path)


def test_load_segment_manifest_rejects_invalid_keyframe_count(tmp_path: Path):
    frame_paths = [tmp_path / f"frame_{idx}.png" for idx in range(4)]
    for path in frame_paths:
        _touch_image(path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "target_duration_seconds": 0.5,
            "segments": [
                {
                    "segment_id": "seg_01",
                    "start_s": 0.0,
                    "end_s": 0.5,
                    "keyframes": [str(path) for path in frame_paths],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="1-3 keyframes"):
        load_segment_manifest(manifest_path)


def test_load_segment_manifest_rejects_invalid_segment_duration(tmp_path: Path):
    frame_path = tmp_path / "frame_01.png"
    _touch_image(frame_path)
    manifest_path = _write_manifest(
        tmp_path,
        {
            "target_duration_seconds": 0.4,
            "segments": [
                {
                    "segment_id": "seg_01",
                    "start_s": 0.0,
                    "end_s": 0.4,
                    "keyframes": [str(frame_path)],
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="exactly 0.5s"):
        load_segment_manifest(manifest_path)
