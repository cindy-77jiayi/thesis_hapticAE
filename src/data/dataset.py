"""PyTorch Dataset classes for haptic signal data."""

import torch
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
        segment_tries: int = 30,
        segment_min_energy: float = 5e-4,
        segment_max_resample: int = 5,
        segment_topk_ratio: float = 0.3,
        segment_peak_pick_prob: float = 0.6,
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax
        self.segment_tries = segment_tries
        self.segment_min_energy = segment_min_energy
        self.segment_max_resample = segment_max_resample
        self.segment_topk_ratio = segment_topk_ratio
        self.segment_peak_pick_prob = segment_peak_pick_prob

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
            tries=self.segment_tries,
            min_energy=self.segment_min_energy,
            max_resample=self.segment_max_resample,
            topk_ratio=self.segment_topk_ratio,
            peak_pick_prob=self.segment_peak_pick_prob,
        )
        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
