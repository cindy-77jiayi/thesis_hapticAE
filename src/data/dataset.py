"""PyTorch Dataset classes for audio signal data."""

import torch
from torch.utils.data import Dataset

from .preprocessing import load_segment_energy


class AudioSignalDataset(Dataset):
    """Dataset that loads energy-rich segments from audio files."""

    def __init__(
        self,
        files: list[str],
        T: int = 4000,
        sr_expect: int = 8000,
        global_rms: float = 1.0,
        scale: float = 0.25,
        use_minmax: bool = False,
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax

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
        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
