from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

torch = pytest.importorskip("torch")

from src.data.loaders import build_model
from src.pipelines.semantic_reconstruction import (
    get_runtime,
    load_frozen_model_spec,
    run_manifest_reconstruction,
    run_three_image_reconstruction,
)


class FakeSemanticClient:
    def infer_segment(self, *, prompt: str, keyframe_paths: list[str]) -> dict:
        assert "semantic_controls" in prompt
        assert len(keyframe_paths) >= 1
        return {
            "response": {"status": "ok"},
            "semantic_controls": {
                "frequency": 0.1,
                "intensity": 0.2,
                "envelope_modulation": 0.3,
                "temporal_grouping": 0.4,
                "sharpness": 0.5,
            },
            "rationale": {
                "frequency": "slow visible motion",
                "intensity": "moderate contact",
                "envelope_modulation": "clear buildup",
                "temporal_grouping": "single grouped event",
                "sharpness": "defined transient",
            },
        }


def _write_fake_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")


def _create_frozen_bundle(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "model_type: vae",
                "run_name: test_runtime",
                "data:",
                "  sr: 8000",
                "  T: 64",
                "  scale: 0.25",
                "  use_minmax: false",
                "  train_split: 0.8",
                "model:",
                "  latent_dim: 8",
                "  channels: [8, 16]",
                "  first_kernel: 9",
                "  kernel_size: 5",
                "  activation: leaky_relu",
                "  norm: group",
                "  logvar_clip: [-10.0, 10.0]",
            ]
        ),
        encoding="utf-8",
    )

    from src.utils.config import load_config

    config = load_config(str(config_path))
    model = build_model(config, torch.device("cpu"))
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    pca_dir = tmp_path / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    z = np.random.default_rng(0).normal(size=(32, 8)).astype(np.float32)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=8)),
        ]
    )
    z_pca = pipe.fit_transform(z)
    with open(pca_dir / "pca_pipe.pkl", "wb") as f:
        pickle.dump(pipe, f)
    np.save(pca_dir / "Z_pca.npy", z_pca)

    frozen_manifest_path = tmp_path / "frozen_manifest.json"
    frozen_manifest_path.write_text(
        json.dumps(
            {
                "run_name": "test_runtime",
                "config_path": str(config_path),
                "checkpoint_path": str(checkpoint_path),
                "pca_dir": str(pca_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return frozen_manifest_path


def test_load_frozen_model_spec_rejects_missing_fields(tmp_path: Path):
    manifest_path = tmp_path / "broken_manifest.json"
    manifest_path.write_text(json.dumps({"config_path": "x"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_frozen_model_spec(manifest_path)


def test_runtime_decodes_to_frozen_target_length(tmp_path: Path):
    frozen_manifest = _create_frozen_bundle(tmp_path)

    runtime = get_runtime(frozen_manifest)
    signal = runtime.decode_pc_vector(np.zeros(8, dtype=np.float32))

    assert runtime.spec.target_len == 64
    assert runtime.spec.target_sr == 8000
    assert signal.shape == (64,)


def test_run_three_image_reconstruction_writes_expected_artifacts(tmp_path: Path):
    frozen_manifest = _create_frozen_bundle(tmp_path)
    image_paths = [tmp_path / f"frame_{idx}.png" for idx in range(3)]
    for image_path in image_paths:
        _write_fake_png(image_path)

    run_dir = tmp_path / "run"
    result = run_three_image_reconstruction(
        image_paths=[str(path) for path in image_paths],
        notes="short tactile event",
        run_dir=run_dir,
        frozen_manifest_path=frozen_manifest,
        client=FakeSemanticClient(),
    )

    assert Path(result["generated_wav"]).exists()
    assert Path(result["waveform_image"]).exists()
    assert Path(result["run_dir"], "three_image_request.json").exists()
    assert Path(result["run_dir"], "normalized_segment_manifest.json").exists()
    assert Path(result["run_dir"], "reconstruction_payload.json").exists()
    assert Path(result["run_dir"], "run_metadata.json").exists()
    assert Path(result["run_dir"], "segments", "seg_01", "raw_response.json").exists()
    assert Path(result["run_dir"], "segments", "seg_01", "semantic_controls.json").exists()
    assert Path(result["run_dir"], "segments", "seg_01", "pc_vector.json").exists()
    assert result["segments"][0]["pc_vector"][5:] == [0.0, 0.0, 0.0]


def test_run_manifest_reconstruction_uses_manifest_file(tmp_path: Path):
    frozen_manifest = _create_frozen_bundle(tmp_path)
    image_path = tmp_path / "frame_01.png"
    _write_fake_png(image_path)

    manifest_path = tmp_path / "segment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "target_duration_seconds": 0.5,
                "segments": [
                    {
                        "segment_id": "seg_01",
                        "start_s": 0.0,
                        "end_s": 0.5,
                        "keyframes": [str(image_path)],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = run_manifest_reconstruction(
        manifest_path=manifest_path,
        run_dir=tmp_path / "manifest_run",
        frozen_manifest_path=frozen_manifest,
        client=FakeSemanticClient(),
    )

    assert Path(result["generated_wav"]).exists()
    assert Path(result["waveform_image"]).exists()
    assert result["manifest"]["segments"][0]["segment_id"] == "seg_01"
