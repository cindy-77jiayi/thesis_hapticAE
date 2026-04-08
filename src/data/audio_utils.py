"""Audio loading, channel conversion, and resampling helpers."""

from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa


def ensure_audio_channels(audio: np.ndarray, channels: int = 1) -> np.ndarray:
    """Convert audio arrays of shape (C, T) to the requested channel count."""
    if audio.ndim != 2:
        raise ValueError(f"Expected audio with shape (channels, time), got {audio.shape}")

    src_channels, _ = audio.shape
    if src_channels == channels:
        return audio
    if channels == 1:
        return audio.mean(axis=0, keepdims=True)
    if src_channels == 1:
        return np.repeat(audio, channels, axis=0)
    if src_channels >= channels:
        return audio[:channels]
    raise ValueError(
        "Audio has fewer channels than requested and is not mono; "
        f"got {src_channels}, requested {channels}."
    )


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample an array of shape (C, T) while preserving channel count."""
    if audio.ndim != 2:
        raise ValueError(f"Expected audio with shape (channels, time), got {audio.shape}")
    if from_rate == to_rate:
        return audio

    resampled = [
        librosa.resample(channel, orig_sr=from_rate, target_sr=to_rate)
        for channel in audio
    ]
    return np.stack(resampled, axis=0).astype(np.float32, copy=False)


def load_audio(
    path: str,
    target_sr: int | None = None,
    target_channels: int = 1,
) -> tuple[np.ndarray, int]:
    """Load audio as float32 with shape (C, T), then adapt sample rate/channels."""
    wav, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = np.asarray(wav.T, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = ensure_audio_channels(audio, channels=target_channels)

    if target_sr is not None:
        audio = resample_audio(audio, from_rate=int(sr), to_rate=int(target_sr))
        sr = int(target_sr)

    return audio, int(sr)

