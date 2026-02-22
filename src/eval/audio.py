"""Audio playback utilities for comparing original and reconstructed signals."""

import numpy as np


def prep_audio(w: np.ndarray) -> np.ndarray:
    """Normalize waveform to [-1, 1] for audio playback."""
    w = w.astype(np.float32)
    if w.min() >= 0 and w.max() <= 1.0:
        w = w * 2.0 - 1.0
    m = np.max(np.abs(w)) + 1e-8
    w = w / m
    return np.clip(w, -1.0, 1.0)


def play_ab_comparison(
    x_np: np.ndarray,
    xhat_np: np.ndarray,
    sr: int = 8000,
    n_samples: int | None = None,
):
    """Play A/B comparison audio (original -> silence -> reconstruction).

    Requires IPython.display (works in Jupyter/Colab).
    """
    from IPython.display import Audio, display

    silence = np.zeros(int(0.2 * sr), dtype=np.float32)
    n = n_samples or len(x_np)
    n = min(n, len(x_np))

    for i in range(n):
        orig = prep_audio(x_np[i])
        recon = prep_audio(xhat_np[i])
        print(
            f"--- Sample {i} | "
            f"orig max {np.max(np.abs(x_np[i])):.4f} | "
            f"recon max {np.max(np.abs(xhat_np[i])):.4f} ---"
        )
        ab = np.concatenate([orig, silence, recon])
        display(Audio(ab, rate=sr))
