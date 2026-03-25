"""Data augmentation helpers for waveform-level training perturbations."""

import numpy as np


def sample_mixing_gain(g_min: float, g_max: float) -> float:
    """Sample a uniform mixing gain g in [g_min, g_max]."""
    if g_min > g_max:
        raise ValueError(f"mixing_gain_min ({g_min}) must be <= mixing_gain_max ({g_max})")
    return float(np.random.uniform(g_min, g_max))


def apply_zero_padded_offset(x: np.ndarray, offset: int) -> np.ndarray:
    """Shift waveform with zero padding (no circular wrap) while keeping length."""
    y = np.zeros_like(x)
    T = x.shape[0]
    if offset == 0:
        return x.copy()
    if offset > 0:
        if offset < T:
            y[offset:] = x[: T - offset]
    else:
        shift = -offset
        if shift < T:
            y[: T - shift] = x[shift:]
    return y


def mix_waveforms(x1: np.ndarray, x2_shifted: np.ndarray, g: float) -> np.ndarray:
    """Blend two waveforms using x_mix = (1 - g) * x1 + g * x2_shifted."""
    g32 = np.float32(g)
    return (np.float32(1.0) - g32) * x1 + g32 * x2_shifted


def soft_peak_normalize_and_clip(
    x_mix: np.ndarray,
    enabled: bool,
    peak_target: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Soft-peak normalize only on overflow, then apply final safety clipping."""
    peak = float(np.max(np.abs(x_mix))) if x_mix.size > 0 else 0.0
    if enabled and peak > peak_target and peak > 0.0:
        x_mix = x_mix * (peak_target / peak)
    return np.clip(x_mix, clip_min, clip_max)
