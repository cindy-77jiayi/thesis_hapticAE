"""Shared runtime for semantic-control inference and local VAE reconstruction."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from src.llm import (
    DEFAULT_OPENAI_MODEL,
    OpenAISemanticSegmentClient,
    build_reconstruction_payload,
    build_segment_prompt,
    load_segment_manifest,
    validate_segment_manifest,
)
from src.llm.manifest import SEGMENT_DURATION_SECONDS
from src.semantic.mapping import semantic_to_pca
from src.semantic.pc_semantics import CANONICAL_SEMANTIC_ORDER, load_semantic_schema
from src.utils.config import load_config


def _load_torch_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch is required for local VAE reconstruction. Install `torch` to decode waveforms."
        ) from exc

    from src.data.loaders import build_model, load_checkpoint

    return torch, build_model, load_checkpoint


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _serialize_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return {"raw_response": str(response)}


def _resolve_manifest_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def create_session_run_dir(base_dir: str | Path = "outputs/vae_llm_ui") -> Path:
    """Create one timestamped run directory for a UI or CLI reconstruction session."""
    root = Path(base_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_waveform_plot(signal: np.ndarray, sr: int, path: str | Path) -> str:
    """Save a single waveform preview image."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    t = np.arange(signal.size, dtype=np.float32) / float(sr)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t, signal, linewidth=0.9, color="steelblue")
    ax.set_title("Generated Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(float(t[0]) if t.size else 0.0, float(t[-1]) if t.size else 0.5)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def save_audio(signal: np.ndarray, sr: int, path: str | Path) -> str:
    """Persist one mono waveform to disk as a wav file."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(signal, dtype=np.float32).reshape(-1)
    sf.write(target, audio, int(sr))
    return str(target)


@dataclass(frozen=True)
class FrozenModelSpec:
    """Resolved runtime specification for one frozen VAE + PCA bundle."""

    manifest_path: Path
    config_path: Path
    checkpoint_path: Path
    pca_dir: Path
    run_name: str
    target_sr: int
    target_len: int


class SemanticReconstructionRuntime:
    """Load one frozen VAE + PCA bundle and decode PCA controls locally."""

    def __init__(self, spec: FrozenModelSpec) -> None:
        torch, build_model, load_checkpoint = _load_torch_runtime()
        self.spec = spec
        self.config = load_config(str(spec.config_path))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        self.model = build_model(self.config, self.device)
        load_checkpoint(self.model, str(spec.checkpoint_path), self.device)

        pipe_path = spec.pca_dir / "pca_pipe.pkl"
        if not pipe_path.exists():
            raise FileNotFoundError(f"PCA pipe not found: {pipe_path}")
        with open(pipe_path, "rb") as f:
            self.pipe = pickle.load(f)

    def decode_pc_vector(self, pc_vector: list[float] | np.ndarray) -> np.ndarray:
        """Decode one PCA vector into a waveform segment."""
        vector = np.asarray(pc_vector, dtype=np.float32).reshape(-1)
        latent = self.pipe.inverse_transform(vector.reshape(1, -1)).reshape(-1)
        latent_t = self._torch.from_numpy(np.asarray(latent, dtype=np.float32)).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            signal = self.model.decode(latent_t, target_len=self.spec.target_len).squeeze().detach().cpu().numpy()
        return np.asarray(signal, dtype=np.float32).reshape(-1)

    def decode_segments(self, segment_outputs: list[dict[str, Any]]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Decode every segment and concatenate them into one final waveform."""
        segment_signals = [self.decode_pc_vector(segment["pc_vector"]) for segment in segment_outputs]
        if not segment_signals:
            raise ValueError("segment_outputs must not be empty.")
        full_signal = np.concatenate(segment_signals, axis=0).astype(np.float32, copy=False)
        return full_signal, segment_signals


_RUNTIME_CACHE: dict[str, SemanticReconstructionRuntime] = {}


def load_frozen_model_spec(frozen_manifest_path: str | Path) -> FrozenModelSpec:
    """Resolve and validate one frozen runtime manifest."""
    manifest_path = Path(frozen_manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frozen manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    base_dir = manifest_path.parent
    required = ["config_path", "checkpoint_path", "pca_dir"]
    missing = [key for key in required if key not in payload or not str(payload[key]).strip()]
    if missing:
        raise ValueError(f"Frozen manifest is missing required fields: {missing}")

    config_path = _resolve_manifest_path(payload["config_path"], base_dir)
    checkpoint_path = _resolve_manifest_path(payload["checkpoint_path"], base_dir)
    pca_dir = _resolve_manifest_path(payload["pca_dir"], base_dir)
    if not pca_dir.is_dir():
        raise FileNotFoundError(f"PCA directory not found: {pca_dir}")

    config = load_config(str(config_path))
    run_name = str(payload.get("run_name") or config.get("run_name") or manifest_path.stem)
    target_sr = int(config["data"]["sr"])
    target_len = int(config["data"]["T"])

    return FrozenModelSpec(
        manifest_path=manifest_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        pca_dir=pca_dir,
        run_name=run_name,
        target_sr=target_sr,
        target_len=target_len,
    )


def get_runtime(frozen_manifest_path: str | Path) -> SemanticReconstructionRuntime:
    """Load and cache the runtime for one frozen manifest path."""
    resolved = str(Path(frozen_manifest_path).resolve())
    runtime = _RUNTIME_CACHE.get(resolved)
    if runtime is not None:
        return runtime

    spec = load_frozen_model_spec(resolved)
    runtime = SemanticReconstructionRuntime(spec)
    _RUNTIME_CACHE[resolved] = runtime
    return runtime


def copy_three_images(image_paths: list[str], run_dir: str | Path) -> list[str]:
    """Copy the 3 ordered image inputs into the run directory."""
    target_dir = Path(run_dir).resolve()
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    copied_images: list[str] = []
    for index, image_path in enumerate(image_paths, start=1):
        source = Path(image_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Input image not found: {source}")
        target = images_dir / f"image_{index:02d}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied_images.append(str(target))
    return copied_images


def build_three_image_manifest(image_paths: list[str], *, notes: str = "") -> dict[str, Any]:
    """Build a single-segment manifest from 3 ordered images."""
    if len(image_paths) != 3:
        raise ValueError(f"Expected exactly 3 images, got {len(image_paths)}.")

    return {
        "target_duration_seconds": SEGMENT_DURATION_SECONDS,
        "segments": [
            {
                "segment_id": "seg_01",
                "start_s": 0.0,
                "end_s": SEGMENT_DURATION_SECONDS,
                "keyframes": image_paths,
                "notes": notes,
            }
        ],
    }


def _run_inference_from_manifest_payload(
    *,
    manifest: dict[str, Any],
    run_dir: str | Path,
    frozen_manifest_path: str | Path,
    prompt_template: str | Path = "llm/prompt_template.md",
    model: str = DEFAULT_OPENAI_MODEL,
    client: OpenAISemanticSegmentClient | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    normalized_manifest = validate_segment_manifest(manifest)
    _write_json(run_path / "normalized_segment_manifest.json", normalized_manifest)

    semantic_schema = load_semantic_schema()
    segment_client = client or OpenAISemanticSegmentClient(model=model)
    runtime = get_runtime(frozen_manifest_path)

    segment_outputs: list[dict[str, Any]] = []
    for segment in normalized_manifest["segments"]:
        segment_dir = run_path / "segments" / segment["segment_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)

        prompt = build_segment_prompt(prompt_template, normalized_manifest, segment)
        result = segment_client.infer_segment(prompt=prompt, keyframe_paths=segment["keyframes"])
        semantic_controls = result["semantic_controls"]
        rationale = result["rationale"]
        pc_vector = semantic_to_pca(semantic_controls, schema=semantic_schema).tolist()

        _write_json(segment_dir / "raw_response.json", _serialize_response(result["response"]))
        _write_json(
            segment_dir / "semantic_controls.json",
            {
                "segment_id": segment["segment_id"],
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
                "rationale": {key: str(rationale[key]) for key in CANONICAL_SEMANTIC_ORDER},
            },
        )
        _write_json(
            segment_dir / "pc_vector.json",
            {
                "segment_id": segment["segment_id"],
                "pc_vector": [float(value) for value in pc_vector],
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
            },
        )

        segment_outputs.append(
            {
                "segment_id": segment["segment_id"],
                "semantic_controls": {
                    key: float(semantic_controls[key]) for key in CANONICAL_SEMANTIC_ORDER
                },
                "pc_vector": [float(value) for value in pc_vector],
                "rationale": {key: str(rationale[key]) for key in CANONICAL_SEMANTIC_ORDER},
            }
        )

    payload = build_reconstruction_payload(
        segment_outputs,
        run_name=runtime.spec.run_name,
        target_sr=runtime.spec.target_sr,
        segment_duration_s=SEGMENT_DURATION_SECONDS,
    )
    _write_json(run_path / "reconstruction_payload.json", payload)

    full_signal, segment_signals = runtime.decode_segments(segment_outputs)
    generated_wav = save_audio(full_signal, runtime.spec.target_sr, run_path / "generated.wav")
    waveform_image = save_waveform_plot(full_signal, runtime.spec.target_sr, run_path / "waveform.png")

    run_metadata = {
        "frozen_manifest": str(runtime.spec.manifest_path),
        "config_path": str(runtime.spec.config_path),
        "checkpoint_path": str(runtime.spec.checkpoint_path),
        "pca_dir": str(runtime.spec.pca_dir),
        "run_name": runtime.spec.run_name,
        "openai_model": model,
        "target_sr": runtime.spec.target_sr,
        "target_len": runtime.spec.target_len,
        "segment_count": len(segment_outputs),
    }
    _write_json(run_path / "run_metadata.json", run_metadata)

    return {
        "run_dir": str(run_path),
        "manifest": normalized_manifest,
        "segments": segment_outputs,
        "reconstruction_payload": payload,
        "generated_wav": generated_wav,
        "waveform_image": waveform_image,
        "full_signal": full_signal,
        "segment_signals": segment_signals,
        "run_metadata": run_metadata,
    }


def run_manifest_reconstruction(
    *,
    manifest_path: str | Path,
    run_dir: str | Path,
    frozen_manifest_path: str | Path,
    prompt_template: str | Path = "llm/prompt_template.md",
    model: str = DEFAULT_OPENAI_MODEL,
    client: OpenAISemanticSegmentClient | None = None,
) -> dict[str, Any]:
    """Run the full manifest -> OpenAI -> semantic -> PC -> VAE decode pipeline."""
    manifest = load_segment_manifest(manifest_path)
    return _run_inference_from_manifest_payload(
        manifest=manifest,
        run_dir=run_dir,
        frozen_manifest_path=frozen_manifest_path,
        prompt_template=prompt_template,
        model=model,
        client=client,
    )


def run_three_image_reconstruction(
    *,
    image_paths: list[str],
    notes: str,
    run_dir: str | Path,
    frozen_manifest_path: str | Path,
    prompt_template: str | Path = "llm/prompt_template.md",
    model: str = DEFAULT_OPENAI_MODEL,
    client: OpenAISemanticSegmentClient | None = None,
) -> dict[str, Any]:
    """Run the 3-image UI flow end to end inside one run directory."""
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)

    copied_images = copy_three_images(image_paths, run_path)
    request_payload = {
        "image_paths": copied_images,
        "notes": notes,
    }
    _write_json(run_path / "three_image_request.json", request_payload)

    manifest = build_three_image_manifest(copied_images, notes=notes)
    return _run_inference_from_manifest_payload(
        manifest=manifest,
        run_dir=run_path,
        frozen_manifest_path=frozen_manifest_path,
        prompt_template=prompt_template,
        model=model,
        client=client,
    )
