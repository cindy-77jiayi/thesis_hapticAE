"""Quantitative signal metrics for characterizing haptic waveforms."""

import numpy as np
from scipy.signal import hilbert


def rms_energy(x: np.ndarray) -> float:
    """Root-mean-square energy of the signal."""
    return float(np.sqrt(np.mean(x ** 2)))


def peak_amplitude(x: np.ndarray) -> float:
    """Maximum absolute amplitude."""
    return float(np.max(np.abs(x)))


def spectral_centroid(x: np.ndarray, sr: int = 8000) -> float:
    """Frequency-domain center of mass (Hz).

    Higher values indicate brighter/sharper signals.
    """
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = X.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * X) / total)


def high_freq_ratio(x: np.ndarray, sr: int = 8000, cutoff_hz: float = 1000.0) -> float:
    """Fraction of spectral energy above cutoff_hz.

    Higher values indicate more high-frequency content (rougher texture).
    """
    X = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = X.sum()
    if total < 1e-12:
        return 0.0
    hf_energy = X[freqs >= cutoff_hz].sum()
    return float(hf_energy / total)


def envelope_decay_slope(x: np.ndarray, sr: int = 8000) -> float:
    """Slope of the log-amplitude envelope (dB/s).

    More negative = faster decay. Near zero = sustained.
    Uses the analytic signal envelope with a smoothing window.
    """
    env = np.abs(hilbert(x))
    win = max(sr // 50, 16)
    if len(env) < win:
        return 0.0
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="valid")

    env_db = 20 * np.log10(env_smooth + 1e-10)
    t = np.arange(len(env_db)) / sr

    if len(t) < 2:
        return 0.0
    coeffs = np.polyfit(t, env_db, deg=1)
    return float(coeffs[0])


def onset_density(x: np.ndarray, sr: int = 8000, threshold_factor: float = 2.0) -> float:
    """Number of amplitude onsets per second.

    Detects points where the short-term energy exceeds a threshold
    relative to the running average.
    """
    frame_len = max(sr // 100, 32)
    hop = frame_len // 2
    n_frames = max(1, (len(x) - frame_len) // hop)

    frame_energy = np.array([
        np.sqrt(np.mean(x[i * hop: i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    if len(frame_energy) < 3:
        return 0.0

    mean_e = np.mean(frame_energy)
    if mean_e < 1e-10:
        return 0.0

    threshold = mean_e * threshold_factor
    above = frame_energy > threshold

    onsets = np.sum(np.diff(above.astype(int)) == 1)
    duration_s = len(x) / sr
    return float(onsets / duration_s) if duration_s > 0 else 0.0


def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio. Higher = more impulsive."""
    rms = rms_energy(x)
    if rms < 1e-10:
        return 0.0
    return float(peak_amplitude(x) / rms)


def compute_all_metrics(x: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """Compute all signal metrics for a waveform."""
    return {
        "rms_energy": rms_energy(x),
        "peak_amplitude": peak_amplitude(x),
        "spectral_centroid_hz": spectral_centroid(x, sr),
        "high_freq_ratio": high_freq_ratio(x, sr),
        "envelope_decay_slope_dBps": envelope_decay_slope(x, sr),
        "onset_density_ps": onset_density(x, sr),
        "crest_factor": crest_factor(x),
    }
