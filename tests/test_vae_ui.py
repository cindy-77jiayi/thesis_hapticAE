from __future__ import annotations

from pathlib import Path

import gradio as gr
import pytest

from src.ui import build_demo
from src.ui.vae_pipeline import run_three_image_generation


def test_run_three_image_generation_requires_exactly_three_images(tmp_path: Path):
    with pytest.raises(gr.Error, match="exactly 3 images"):
        run_three_image_generation(
            image_paths=[str(tmp_path / "a.png"), None, None],
            notes="",
            frozen_manifest="frozen.json",
            model="gpt-4.1-mini",
            outputs_dir=str(tmp_path / "runs"),
            pipeline_runner=lambda **_: {},
        )


def test_run_three_image_generation_uses_runner(tmp_path: Path):
    image_paths = [tmp_path / f"frame_{idx}.png" for idx in range(3)]
    for image_path in image_paths:
        image_path.write_bytes(b"fake")

    captured: dict = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return {
            "segments": [
                {
                    "semantic_controls": {"frequency": 0.1},
                    "rationale": {"frequency": "test"},
                    "pc_vector": [0.0] * 8,
                }
            ],
            "generated_wav": str(tmp_path / "out.wav"),
            "waveform_image": str(tmp_path / "waveform.png"),
            "run_dir": str(tmp_path / "run"),
        }

    result = run_three_image_generation(
        image_paths=[str(path) for path in image_paths],
        notes="demo",
        frozen_manifest="frozen.json",
        model="gpt-4.1-mini",
        outputs_dir=str(tmp_path / "runs"),
        pipeline_runner=fake_runner,
    )

    assert captured["notes"] == "demo"
    assert captured["frozen_manifest_path"] == "frozen.json"
    assert captured["model"] == "gpt-4.1-mini"
    assert result["run_dir"] == str(tmp_path / "run")


def test_build_demo_smoke():
    demo = build_demo(
        frozen_manifest="frozen.json",
        pipeline_runner=lambda **_: {
            "segments": [
                {
                    "semantic_controls": {"frequency": 0.1},
                    "rationale": {"frequency": "test"},
                    "pc_vector": [0.0] * 8,
                }
            ],
            "generated_wav": "generated.wav",
            "waveform_image": "waveform.png",
            "run_dir": "run",
        },
    )

    assert isinstance(demo, gr.Blocks)
