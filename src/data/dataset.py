"""PyTorch Dataset classes for audio signal data."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import load_segment_energy, load_segment_hapticgen


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
        segment_mode: str = "energy",
        random_seek: bool = True,
        sample_with_replacement: bool = False,
        num_samples: int | None = None,
        seed: int = 42,
        min_segment_ratio: float = 1.0,
        normalize_mode: str = "global_rms",
        clip_range: tuple[float, float] | None = (-3.0, 3.0),
    ):
        self.files = files
        self.T = T
        self.sr_expect = sr_expect
        self.global_rms = global_rms
        self.scale = scale
        self.use_minmax = use_minmax
        self.segment_mode = segment_mode
        self.random_seek = random_seek
        self.sample_with_replacement = sample_with_replacement
        self.num_samples = num_samples or len(files)
        self.seed = seed
        self.min_segment_ratio = min_segment_ratio
        self.normalize_mode = normalize_mode
        self.clip_range = clip_range

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.sample_with_replacement:
            rng = np.random.default_rng(self.seed + idx)
            file_idx = int(rng.integers(0, len(self.files)))
        else:
            file_idx = idx % len(self.files)

        path = self.files[file_idx]
        if self.segment_mode == "energy":
            x = load_segment_energy(
                path,
                T=self.T,
                sr_expect=self.sr_expect,
                global_rms=self.global_rms,
                scale=self.scale,
                use_minmax=self.use_minmax,
            )
        elif self.segment_mode == "hapticgen":
            x = load_segment_hapticgen(
                path,
                T=self.T,
                sr_expect=self.sr_expect,
                normalize_mode=self.normalize_mode,
                global_rms=self.global_rms,
                scale=self.scale,
                use_minmax=self.use_minmax,
                random_seek=self.random_seek,
                seed=self.seed + idx,
                min_segment_ratio=self.min_segment_ratio,
                clip_range=self.clip_range,
            )
        else:
            raise ValueError(f"Unknown segment_mode: {self.segment_mode}")
        return torch.from_numpy(x).unsqueeze(0)  # (1, T)
