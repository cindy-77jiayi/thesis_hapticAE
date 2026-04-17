"""Data preprocessing utilities for haptic WAV signals."""

import glob
import json
import os
import random

import numpy as np

from .audio_utils import load_audio


ACCEPTED_MODELS = {"HapticGen"}


def collect_clean_wavs(
    root: str,
    accepted_models: set[str] | None = None,
    accepted_votes: set[int] | None = None,
    include_subdirs: set[str] | None = None,
) -> list[str]:
    """Discover WAV files that pass HapticGen quality filters.

    Scans all subdirectories (expertvoted/, uservoted/, etc.) for .am1.json
    metadata and selects WAVs where model is in accepted_models and vote == 1.

    Args:
        root: Root directory of the hapticgen-dataset repo.
        accepted_models: Set of model names to accept.
            Defaults to ACCEPTED_MODELS = {"HapticGen"} (fine-tuned model only).
        accepted_votes: Set of vote labels to accept.
            Defaults to {1}.
        include_subdirs: Optional subset of first-level dataset subdirs
            (e.g., {"expertvoted"}, {"uservoted"}). If None, scan all.
    """
    if accepted_models is None:
        accepted_models = ACCEPTED_MODELS
    if accepted_votes is None:
        accepted_votes = {1}
    if include_subdirs is not None:
        include_subdirs = {s.lower() for s in include_subdirs}

    wavs = []
    for meta_path in glob.glob(os.path.join(root, "**/*.am1.json"), recursive=True):
        rel = os.path.relpath(meta_path, root)
        first = rel.split(os.sep, 1)[0].lower()
        if include_subdirs is not None and first not in include_subdirs:
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("model") in accepted_models and meta.get("vote") in accepted_votes:
            wav_path = os.path.join(os.path.dirname(meta_path), meta["filename"])
            if os.path.exists(wav_path):
                wavs.append(wav_path)
    return wavs


def estimate_global_rms(
    files: list[str], n: int = 200, sr_expect: int = 8000
) -> float:
    """Estimate median RMS across a random subset of files."""
    picks = random.sample(files, min(n, len(files)))
    rms_values = []
    for p in picks:
        y, _ = load_audio(p, target_sr=sr_expect, target_channels=1)
        y = y[0]
        y = y - np.mean(y)
        rms_values.append(np.sqrt(np.mean(y**2)) + 1e-8)
    return float(np.median(rms_values))


def minmax_norm(seg: np.ndarray) -> np.ndarray:
    """Normalize segment to [0, 1]."""
    mn, mx = seg.min(), seg.max()
    if mx - mn < 1e-8:
        return np.zeros_like(seg)
    return (seg - mn) / (mx - mn)


def augment_segment(
    seg: np.ndarray,
    *,
    gain_range: tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.0,
    shift_max: int = 0,
    dropout_prob: float = 0.0,
    dropout_width: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply lightweight waveform augmentation to a segment."""
    rng = rng or np.random.default_rng()
    out = np.asarray(seg, dtype=np.float32).copy()

    if gain_range[0] > 0 and gain_range[1] > 0:
        gain = float(rng.uniform(gain_range[0], gain_range[1]))
        out *= gain

    if noise_std > 0:
        out += rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    if shift_max > 0:
        shift = int(rng.integers(-shift_max, shift_max + 1))
        if shift != 0:
            out = np.roll(out, shift)

    if dropout_prob > 0 and dropout_width > 0 and rng.random() < dropout_prob:
        width = min(int(dropout_width), len(out))
        start = int(rng.integers(0, len(out) - width + 1))
        out[start : start + width] = 0.0

    return out.astype(np.float32)


def load_segment_energy(
    path: str,
    T: int = 4000,
    sr_expect: int = 8000,
    global_rms: float = 1.0,
    scale: float = 0.25,
    use_minmax: bool = False,
    tries: int = 30,
    min_energy: float = 5e-4,
    max_resample: int = 5,
    clip_range: tuple[float, float] = (-3.0, 3.0),
    search_window_seconds: float | None = None,
    top_k: int = 4,
    random_segment_prob: float = 0.0,
    augment: bool = False,
    augmentation_config: dict | None = None,
) -> np.ndarray:
    """Load a WAV file and extract an energy-rich segment of length T.

    The segment is RMS-normalized, scaled, optionally augmented, and clipped.
    """
    y, _ = load_audio(path, target_sr=sr_expect, target_channels=1)
    y = y[0]

    best_seg_global = None
    best_energy_global = -1.0
    search_window_T = None
    if search_window_seconds is not None:
        search_window_T = max(T, int(float(search_window_seconds) * sr_expect))

    for _ in range(max_resample):
        y_view = y
        if search_window_T is not None and len(y) > search_window_T:
            start = int(np.random.randint(0, len(y) - search_window_T + 1))
            y_view = y[start : start + search_window_T]

        if len(y_view) < T:
            y_view = np.pad(y_view, (0, T - len(y_view)))

        max_start = len(y_view) - T
        candidates: list[tuple[float, np.ndarray]] = []

        for _ in range(tries):
            start = int(np.random.randint(0, max_start + 1))
            seg = y_view[start : start + T]
            seg = seg - np.mean(seg)
            e = float(np.mean(seg**2))
            candidates.append((e, seg))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        top_count = max(1, min(int(top_k), len(candidates)))
        if random_segment_prob > 0 and np.random.random() < random_segment_prob:
            best_energy, best_seg = candidates[int(np.random.randint(0, len(candidates)))]
        else:
            best_energy, best_seg = candidates[int(np.random.randint(0, top_count))]

        if best_energy > best_energy_global:
            best_energy_global = best_energy
            best_seg_global = best_seg

        if best_energy >= min_energy:
            break

    if best_seg_global is None:
        return np.zeros(T, dtype=np.float32)

    seg = best_seg_global / (global_rms + 1e-6)
    seg = seg * scale

    if augment:
        aug_cfg = augmentation_config or {}
        seg = augment_segment(
            seg,
            gain_range=tuple(aug_cfg.get("gain_range", [0.9, 1.1])),
            noise_std=float(aug_cfg.get("noise_std", 0.0)),
            shift_max=int(aug_cfg.get("shift_max", 0)),
            dropout_prob=float(aug_cfg.get("dropout_prob", 0.0)),
            dropout_width=int(aug_cfg.get("dropout_width", 0)),
        )

    seg = np.clip(seg, clip_range[0], clip_range[1])

    if use_minmax:
        seg = minmax_norm(seg)

    return seg.astype(np.float32)
