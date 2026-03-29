"""Data preprocessing utilities for haptic WAV signals."""

import glob
import json
import os
import random

import librosa
import numpy as np


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

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if meta.get("model") in accepted_models and meta.get("vote") in accepted_votes:
            wav_name = meta.get("filename")
            if not wav_name:
                continue
            wav_path = os.path.join(os.path.dirname(meta_path), wav_name)
            if os.path.exists(wav_path):
                wavs.append(wav_path)
    # Keep deterministic order and avoid duplicates when combining roots.
    return sorted(set(wavs))


def estimate_global_rms(
    files: list[str], n: int = 200, sr_expect: int = 8000
) -> float:
    """Estimate median RMS across a random subset of files."""
    picks = random.sample(files, min(n, len(files)))
    rms_values = []
    for p in picks:
        y, sr = librosa.load(p, sr=None, mono=True)
        if sr != sr_expect:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_expect)
        y = y - np.mean(y)
        rms_values.append(np.sqrt(np.mean(y**2)) + 1e-8)
    return float(np.median(rms_values))


def minmax_norm(seg: np.ndarray) -> np.ndarray:
    """Normalize segment to [0, 1]."""
    mn, mx = seg.min(), seg.max()
    if mx - mn < 1e-8:
        return np.zeros_like(seg)
    return (seg - mn) / (mx - mn)


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
    topk_ratio: float = 0.3,
    peak_pick_prob: float = 0.6,
    clip_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    """Load a WAV file and extract an energy-rich segment of length T.

    Candidate windows are sampled randomly, then selected from the highest-energy
    subset. This keeps training focused on informative content while preserving
    non-peak regions (e.g. decay tails).

    The chosen segment is RMS-normalized, scaled, and clipped to *clip_range*.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != sr_expect:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_expect)

    if len(y) < T:
        y = np.pad(y, (0, T - len(y)))

    max_start = len(y) - T
    best_seg_global = None
    best_energy_global = -1.0

    tries = max(1, int(tries))
    topk_ratio = float(np.clip(topk_ratio, 0.0, 1.0))
    peak_pick_prob = float(np.clip(peak_pick_prob, 0.0, 1.0))

    for _ in range(max_resample):
        candidates: list[tuple[float, np.ndarray]] = []

        for _ in range(tries):
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            seg = y[start : start + T]
            seg = seg - np.mean(seg)
            e = float(np.mean(seg**2))
            candidates.append((e, seg))

        candidates.sort(key=lambda item: item[0])
        topk = max(1, int(np.ceil(len(candidates) * topk_ratio)))
        top_pool = candidates[-topk:]
        if np.random.rand() < peak_pick_prob:
            picked_energy, picked_seg = top_pool[-1]
        else:
            ridx = np.random.randint(0, len(top_pool))
            picked_energy, picked_seg = top_pool[ridx]

        if picked_energy > best_energy_global:
            best_energy_global = picked_energy
            best_seg_global = picked_seg

        if picked_energy >= min_energy:
            break

    if best_seg_global is None:
        return np.zeros(T, dtype=np.float32)

    seg = best_seg_global / (global_rms + 1e-6)
    seg = seg * scale
    seg = np.clip(seg, clip_range[0], clip_range[1])

    if use_minmax:
        seg = minmax_norm(seg)

    return seg.astype(np.float32)
