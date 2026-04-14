from __future__ import annotations

import json
from pathlib import Path

from src.llm_hapticgen.artifacts import create_session_run_dir, import_colab_outputs, save_generation_artifacts


def test_save_generation_artifacts_copies_images_and_writes_files(tmp_path: Path):
    image_paths = []
    for idx in range(3):
        image_path = tmp_path / f"input_{idx}.png"
        image_path.write_bytes(b"fake")
        image_paths.append(str(image_path))

    run_dir = create_session_run_dir(tmp_path / "runs")
    result = save_generation_artifacts(
        run_dir=run_dir,
        image_paths=image_paths,
        notes="test notes",
        openai_payload={
            "visual_summary": "summary",
            "haptic_prompt": "prompt",
            "negative_prompt": "",
            "rationale": "why",
        },
    )

    assert Path(result["run_dir"]).exists()
    assert Path(result["request_json"]).exists()
    assert Path(result["response_json"]).exists()
    assert Path(result["prompt_txt"]).read_text(encoding="utf-8") == "prompt"

    request_payload = json.loads(Path(result["request_json"]).read_text(encoding="utf-8"))
    assert len(request_payload["image_paths"]) == 3


def test_import_colab_outputs_copies_result_files(tmp_path: Path):
    run_dir = create_session_run_dir(tmp_path / "runs")
    wav_path = tmp_path / "generated.wav"
    wav_path.write_bytes(b"wav")
    waveform_path = tmp_path / "waveform.png"
    waveform_path.write_bytes(b"png")

    metadata = import_colab_outputs(
        run_dir=run_dir,
        generated_wav=wav_path,
        waveform_image=waveform_path,
    )

    assert Path(str(metadata["generated_wav"])).exists()
    assert Path(str(metadata["waveform_image"])).exists()
