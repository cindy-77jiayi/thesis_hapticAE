"""Data preprocessing utilities for haptic WAV signals."""

import glob
import json
import os
import random

import librosa
import numpy as np


def collect_clean_wavs(root: str) -> list[str]:
    """Discover WAV files that pass HapticGen quality filters.

    Looks for .am1.json metadata files and selects WAVs where
    model == 'HapticGen-Initial' and vote == 1.
    """
    wavs = []
    for meta_path in glob.glob(os.path.join(root, "**/*.am1.json"), recursive=True):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("model") == "HapticGen-Initial" and meta.get("vote") == 1:
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
) -> np.ndarray:
    """Load a WAV file and extract an energy-rich segment of length T.

    The segment is RMS-normalized, scaled, and clipped to [-3, 3].
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != sr_expect:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_expect)

    if len(y) < T:
        y = np.pad(y, (0, T - len(y)))

    max_start = len(y) - T
    best_seg_global = None
    best_energy_global = -1.0

    for _ in range(max_resample):
        best_seg = None
        best_energy = -1.0

        for _ in range(tries):
            start = np.random.randint(0, max_start + 1)
            seg = y[start : start + T]
            seg = seg - np.mean(seg)
            e = float(np.mean(seg**2))

            if e > best_energy:
                best_energy = e
                best_seg = seg

        if best_energy > best_energy_global:
            best_energy_global = best_energy
            best_seg_global = best_seg

        if best_energy >= min_energy:
            break

    if best_seg_global is None:
        return np.zeros(T, dtype=np.float32)

    seg = best_seg_global / (global_rms + 1e-6)
    seg = seg * scale
    seg = np.clip(seg, -3.0, 3.0)

    if use_minmax:
        seg = minmax_norm(seg)

    return seg.astype(np.float32)


def make_vibration(
    sr: int = 1000, duration: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """Generate a synthetic vibrotactile signal for testing."""
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = np.random.uniform(80, 300)
    signal = np.sin(2 * np.pi * freq * t)

    decay = np.exp(-np.random.uniform(1, 5) * t)
    signal *= decay

    n_pulses = np.random.randint(2, 6)
    for _ in range(n_pulses):
        pos = np.random.randint(0, len(t))
        width = np.random.randint(5, 30)
        amp = np.random.uniform(0.3, 1.0)
        lo = max(0, pos - width)
        hi = min(len(t), pos + width)
        signal[lo:hi] += amp

    signal += np.random.randn(len(t)) * 0.02

    mx = np.max(np.abs(signal)) + 1e-8
    signal = signal / mx
    return signal.astype(np.float32)
