"""Artifact helpers for the 3-image HapticGen validation flow."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def create_session_run_dir(base_dir: str | Path = "outputs/hapticgen_llm_ui") -> Path:
    """Create one timestamped run directory for a UI generation session."""
    root = Path(base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_generation_artifacts(
    *,
    run_dir: str | Path,
    image_paths: list[str],
    notes: str,
    openai_payload: dict[str, Any],
) -> dict[str, str]:
    """Persist copied inputs and OpenAI outputs for one UI run."""
    target_dir = Path(run_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_images: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        source = Path(image_path).resolve()
        target = images_dir / f"image_{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied_images.append(str(target))

    request_payload = {
        "image_paths": copied_images,
        "notes": notes,
    }
    _write_json(target_dir / "three_image_request.json", request_payload)
    _write_json(target_dir / "openai_response.json", openai_payload)
    (target_dir / "hapticgen_prompt.txt").write_text(
        str(openai_payload["haptic_prompt"]).strip(),
        encoding="utf-8",
    )
    return {
        "request_json": str((target_dir / "three_image_request.json").resolve()),
        "response_json": str((target_dir / "openai_response.json").resolve()),
        "prompt_txt": str((target_dir / "hapticgen_prompt.txt").resolve()),
        "run_dir": str(target_dir),
    }


def import_colab_outputs(
    *,
    run_dir: str | Path,
    generated_wav: str | Path | None,
    waveform_image: str | Path | None,
) -> dict[str, str | None]:
    """Copy Colab-side outputs into a local run directory for UI preview."""
    target_dir = Path(run_dir).resolve()
    if not target_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {target_dir}")

    colab_dir = target_dir / "colab_results"
    colab_dir.mkdir(parents=True, exist_ok=True)

    wav_out = None
    if generated_wav:
        source = Path(generated_wav).resolve()
        wav_out = colab_dir / f"generated{source.suffix.lower()}"
        shutil.copy2(source, wav_out)

    waveform_out = None
    if waveform_image:
        source = Path(waveform_image).resolve()
        waveform_out = colab_dir / f"waveform{source.suffix.lower()}"
        shutil.copy2(source, waveform_out)

    metadata = {
        "generated_wav": str(wav_out) if wav_out else None,
        "waveform_image": str(waveform_out) if waveform_out else None,
        "imported_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(colab_dir / "generation_metadata.json", metadata)
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
