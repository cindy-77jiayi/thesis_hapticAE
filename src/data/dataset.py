"""PyTorch Dataset classes for haptic signal data."""

import torch
import numpy as np
from torch.utils.data import Dataset

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
        segment_seed: int | None = None,
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax
        self.segment_seed = segment_seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = None
        if self.segment_seed is not None:
            rng = np.random.default_rng(self.segment_seed + int(idx))

        x = load_segment_energy(
            self.files[idx],
            T=self.T,
            sr_expect=self.sr_expect,
            global_rms=self.global_rms,
            scale=self.scale,
            use_minmax=self.use_minmax,
            rng=rng,
        )
        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
