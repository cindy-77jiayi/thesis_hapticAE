"""Data preprocessing utilities for audio signal datasets."""

from pathlib import Path
import random

import librosa
import numpy as np


DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac")


def collect_audio_files(
    root: str,
    extensions: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    """Discover audio files recursively under a dataset root.

    Args:
        root: Root directory containing dataset audio files.
        extensions: Optional iterable of allowed audio suffixes.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    allowed_exts = extensions or DEFAULT_AUDIO_EXTENSIONS
    normalized_exts = {
        ext.lower() if str(ext).startswith(".") else f".{str(ext).lower()}"
        for ext in allowed_exts
    }

    audio_files = [
        str(path)
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_exts
    ]
    audio_files.sort()
    return audio_files


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
    clip_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
    """Load a WAV file and extract an energy-rich segment of length T.

    The segment is RMS-normalized, scaled, and clipped to *clip_range*.
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
    seg = np.clip(seg, clip_range[0], clip_range[1])

    if use_minmax:
        seg = minmax_norm(seg)

    return seg.astype(np.float32)
