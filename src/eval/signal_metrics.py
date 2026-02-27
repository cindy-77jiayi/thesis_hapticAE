"""Quantitative signal metrics for characterizing haptic waveforms.

Metrics are grouped by the perceptual dimension they target:
  Intensity:   rms_energy, peak_amplitude
  Sustain:     envelope_decay_slope, late_early_energy_ratio
  Warmth:      spectral_centroid, low_high_band_ratio
  Sharpness:   attack_time, high_freq_ratio, crest_factor
  Rhythm:      onset_density, ioi_entropy
  Continuity:  zero_crossing_rate, gap_ratio
  Texture:     short_term_variance, am_modulation_index
"""

import numpy as np
from scipy.signal import hilbert


# ---------------------------------------------------------------------------
# Intensity
# ---------------------------------------------------------------------------

def rms_energy(x: np.ndarray) -> float:
    """Root-mean-square energy of the signal."""
    return float(np.sqrt(np.mean(x ** 2)))


def peak_amplitude(x: np.ndarray) -> float:
    """Maximum absolute amplitude."""
    return float(np.max(np.abs(x)))


# ---------------------------------------------------------------------------
# Sustain / Decay
# ---------------------------------------------------------------------------

def envelope_decay_slope(x: np.ndarray, sr: int = 8000) -> float:
    """Slope of the log-amplitude envelope (dB/s).

    More negative = faster decay. Near zero = sustained.
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


def late_early_energy_ratio(x: np.ndarray) -> float:
    """Ratio of energy in the second half to the first half.

    > 1 = energy builds up or sustains; < 1 = energy decays.
    """
    mid = len(x) // 2
    e_early = np.mean(x[:mid] ** 2) + 1e-12
    e_late = np.mean(x[mid:] ** 2) + 1e-12
    return float(e_late / e_early)


# ---------------------------------------------------------------------------
# Warmth / Brightness
# ---------------------------------------------------------------------------

def spectral_centroid(x: np.ndarray, sr: int = 8000) -> float:
    """Frequency-domain center of mass (Hz). Higher = brighter."""
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = X.sum()
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * X) / total)


def low_high_band_ratio(x: np.ndarray, sr: int = 8000, cutoff_hz: float = 1000.0) -> float:
    """Ratio of low-band to high-band energy. Higher = warmer."""
    X = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    lo = X[freqs < cutoff_hz].sum() + 1e-12
    hi = X[freqs >= cutoff_hz].sum() + 1e-12
    return float(lo / hi)


def high_freq_ratio(x: np.ndarray, sr: int = 8000, cutoff_hz: float = 1000.0) -> float:
    """Fraction of spectral energy above cutoff_hz. Higher = rougher."""
    X = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    total = X.sum()
    if total < 1e-12:
        return 0.0
    return float(X[freqs >= cutoff_hz].sum() / total)


# ---------------------------------------------------------------------------
# Attack / Sharpness
# ---------------------------------------------------------------------------

def attack_time(x: np.ndarray, sr: int = 8000) -> float:
    """Time (seconds) for envelope to rise from 10% to 90% of peak.

    Shorter = sharper attack. Returns signal duration if no clear attack.
    """
    env = np.abs(hilbert(x))
    win = max(sr // 100, 8)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")

    peak = env_smooth.max()
    if peak < 1e-10:
        return float(len(x) / sr)

    thresh_lo = 0.1 * peak
    thresh_hi = 0.9 * peak

    idx_lo = np.argmax(env_smooth >= thresh_lo)
    idx_hi = np.argmax(env_smooth >= thresh_hi)

    if idx_hi <= idx_lo:
        return float(len(x) / sr)
    return float((idx_hi - idx_lo) / sr)


def crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio. Higher = more impulsive."""
    rms = rms_energy(x)
    if rms < 1e-10:
        return 0.0
    return float(peak_amplitude(x) / rms)


# ---------------------------------------------------------------------------
# Rhythm / Event density
# ---------------------------------------------------------------------------

def onset_density(x: np.ndarray, sr: int = 8000, threshold_factor: float = 2.0) -> float:
    """Number of amplitude onsets per second."""
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


def ioi_entropy(x: np.ndarray, sr: int = 8000, threshold_factor: float = 2.0) -> float:
    """Shannon entropy of inter-onset intervals (bits).

    Higher = more irregular/complex rhythm. 0 = perfectly periodic.
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
    onset_idx = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(onset_idx) < 2:
        return 0.0

    iois = np.diff(onset_idx).astype(float)
    iois_norm = iois / iois.sum()
    entropy = -np.sum(iois_norm * np.log2(iois_norm + 1e-12))
    return float(entropy)


# ---------------------------------------------------------------------------
# Continuity
# ---------------------------------------------------------------------------

def zero_crossing_rate(x: np.ndarray, sr: int = 8000) -> float:
    """Zero-crossing rate (crossings per second)."""
    crossings = np.sum(np.abs(np.diff(np.sign(x))) > 0)
    duration_s = len(x) / sr
    return float(crossings / duration_s) if duration_s > 0 else 0.0


def gap_ratio(x: np.ndarray, sr: int = 8000, silence_thresh: float = 0.01) -> float:
    """Fraction of signal below silence threshold (0–1).

    Higher = more gaps/silence. Lower = more continuous.
    """
    env = np.abs(hilbert(x))
    win = max(sr // 100, 8)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="same")

    rms = np.sqrt(np.mean(x ** 2))
    thresh = max(silence_thresh, rms * 0.1)
    silent_samples = np.sum(env_smooth < thresh)
    return float(silent_samples / len(x))


# ---------------------------------------------------------------------------
# Temporal texture
# ---------------------------------------------------------------------------

def short_term_variance(x: np.ndarray, sr: int = 8000) -> float:
    """Variance of short-term RMS energy across frames.

    Higher = more temporal variation in energy.
    """
    frame_len = max(sr // 20, 64)
    hop = frame_len // 2
    n_frames = max(1, (len(x) - frame_len) // hop)

    frame_rms = np.array([
        np.sqrt(np.mean(x[i * hop: i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    if len(frame_rms) < 2:
        return 0.0
    return float(np.var(frame_rms))


def am_modulation_index(x: np.ndarray, sr: int = 8000) -> float:
    """Amplitude modulation depth (0–1).

    Computed as (env_max - env_min) / (env_max + env_min) on the smoothed
    envelope. Higher = more pronounced amplitude modulation.
    """
    env = np.abs(hilbert(x))
    win = max(sr // 50, 16)
    kernel = np.ones(win) / win
    env_smooth = np.convolve(env, kernel, mode="valid")

    if len(env_smooth) < 2:
        return 0.0

    env_max = env_smooth.max()
    env_min = env_smooth.min()
    denom = env_max + env_min
    if denom < 1e-10:
        return 0.0
    return float((env_max - env_min) / denom)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def compute_all_metrics(x: np.ndarray, sr: int = 8000) -> dict[str, float]:
    """Compute all signal metrics for a waveform."""
    return {
        "rms_energy": rms_energy(x),
        "peak_amplitude": peak_amplitude(x),
        "spectral_centroid_hz": spectral_centroid(x, sr),
        "high_freq_ratio": high_freq_ratio(x, sr),
        "low_high_band_ratio": low_high_band_ratio(x, sr),
        "envelope_decay_slope_dBps": envelope_decay_slope(x, sr),
        "late_early_energy_ratio": late_early_energy_ratio(x),
        "attack_time_s": attack_time(x, sr),
        "onset_density_ps": onset_density(x, sr),
        "ioi_entropy_bits": ioi_entropy(x, sr),
        "zero_crossing_rate_ps": zero_crossing_rate(x, sr),
        "gap_ratio": gap_ratio(x, sr),
        "short_term_variance": short_term_variance(x, sr),
        "am_modulation_index": am_modulation_index(x, sr),
        "crest_factor": crest_factor(x),
    }
