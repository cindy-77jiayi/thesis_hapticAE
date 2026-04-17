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
        clip_range: tuple[float, float] = (-3.0, 3.0),
        segment_tries: int = 30,
        min_energy: float = 5e-4,
        max_resample: int = 5,
        search_window_seconds: float | None = None,
        segment_top_k: int = 4,
        random_segment_prob: float = 0.0,
        augment: bool = False,
        augmentation_config: dict | None = None,
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax
        self.clip_range = clip_range
        self.segment_tries = segment_tries
        self.min_energy = min_energy
        self.max_resample = max_resample
        self.search_window_seconds = search_window_seconds
        self.segment_top_k = segment_top_k
        self.random_segment_prob = random_segment_prob
        self.augment = augment
        self.augmentation_config = augmentation_config or {}

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
            clip_range=self.clip_range,
            tries=self.segment_tries,
            min_energy=self.min_energy,
            max_resample=self.max_resample,
            search_window_seconds=self.search_window_seconds,
            top_k=self.segment_top_k,
            random_segment_prob=self.random_segment_prob,
            augment=self.augment,
            augmentation_config=self.augmentation_config,
        )
        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
