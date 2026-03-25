"""PyTorch Dataset classes for haptic signal data."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import (
    apply_zero_padded_offset,
    mix_waveforms,
    sample_mixing_gain,
    soft_peak_normalize_and_clip,
)
from .preprocessing import load_segment_energy


class HapticWavDataset(Dataset):
    """Dataset that loads energy-rich segments from haptic WAV files."""

    def __init__(
        self,
        files: list[str],
        T: int = 4000,
        sr_expect: int = 8000,
        global_rms: float = 1.0,
        scale: float = 0.25,
        use_minmax: bool = False,
        use_mixing_augmentation: bool = False,
        mixing_probability: float = 0.0,
        mixing_gain_min: float = 0.2,
        mixing_gain_max: float = 0.8,
        mixing_offset_enabled: bool = True,
        mixing_offset_min: int = -400,
        mixing_offset_max: int = 400,
        mixing_normalize_enabled: bool = True,
        mixing_normalize_peak_target: float = 3.0,
        mixing_clip_min: float = -3.0,
        mixing_clip_max: float = 3.0,
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax
        self.use_mixing_augmentation = use_mixing_augmentation
        self.mixing_probability = mixing_probability
        self.mixing_gain_min = mixing_gain_min
        self.mixing_gain_max = mixing_gain_max
        self.mixing_offset_enabled = mixing_offset_enabled
        self.mixing_offset_min = mixing_offset_min
        self.mixing_offset_max = mixing_offset_max
        self.mixing_normalize_enabled = mixing_normalize_enabled
        self.mixing_normalize_peak_target = mixing_normalize_peak_target
        self.mixing_clip_min = mixing_clip_min
        self.mixing_clip_max = mixing_clip_max

        if not (0.0 <= self.mixing_probability <= 1.0):
            raise ValueError("mixing_probability must be in [0, 1]")
        if self.mixing_gain_min > self.mixing_gain_max:
            raise ValueError("mixing_gain_min must be <= mixing_gain_max")
        if self.mixing_offset_min > self.mixing_offset_max:
            raise ValueError("mixing_offset_min must be <= mixing_offset_max")
        if self.mixing_normalize_peak_target <= 0.0:
            raise ValueError("mixing_normalize_peak_target must be > 0")
        if self.mixing_clip_min > self.mixing_clip_max:
            raise ValueError("mixing_clip_min must be <= mixing_clip_max")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = load_segment_energy(
            self.files[idx],
            T=self.T,
            sr_expect=self.sr_expect,
            global_rms=self.global_rms,
            scale=self.scale,
            use_minmax=self.use_minmax,
        )

        # Mixing exposes the encoder/decoder to additive factor composition,
        # which can encourage more linearly separable latent structure for PCA.
        do_mix = (
            self.use_mixing_augmentation
            and len(self.files) > 1
            and np.random.rand() < self.mixing_probability
        )
        if do_mix:
            idx2 = idx
            while idx2 == idx:
                idx2 = int(np.random.randint(0, len(self.files)))

            x2 = load_segment_energy(
                self.files[idx2],
                T=self.T,
                sr_expect=self.sr_expect,
                global_rms=self.global_rms,
                scale=self.scale,
                use_minmax=self.use_minmax,
            )

            offset = 0
            if self.mixing_offset_enabled:
                offset = int(np.random.randint(self.mixing_offset_min, self.mixing_offset_max + 1))
            x2_shifted = apply_zero_padded_offset(x2, offset)

            g = sample_mixing_gain(self.mixing_gain_min, self.mixing_gain_max)
            x_mix = mix_waveforms(x, x2_shifted, g)

            # Soft peak normalization only acts on overflow, preserving natural
            # loudness contrast while preventing unstable clipping explosions.
            x = soft_peak_normalize_and_clip(
                x_mix,
                enabled=self.mixing_normalize_enabled,
                peak_target=self.mixing_normalize_peak_target,
                clip_min=self.mixing_clip_min,
                clip_max=self.mixing_clip_max,
            ).astype(np.float32, copy=False)

        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
