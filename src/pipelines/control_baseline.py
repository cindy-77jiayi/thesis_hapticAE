"""Post-hoc PCA baseline built from pooled codec features."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


def pool_codec_features(features: torch.Tensor) -> torch.Tensor:
    """Pool sequence features into a global control vector via mean and std."""
    mean = features.mean(dim=-1)
    std = features.std(dim=-1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def extract_pooled_features_and_sequences(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled codec features and quantized sequence latents."""
    model.eval()
    pooled_rows = []
    seq_rows = []

    with torch.no_grad():
        for x in tqdm(loader, desc="Extracting pooled features", leave=False):
            x = x.to(device)
            features = model.encode_features(x)
            pooled = pool_codec_features(features)
            z_seq, _, _ = model.quantize_sequence(features)
            pooled_rows.append(pooled.cpu().numpy())
            seq_rows.append(z_seq.flatten(start_dim=1).cpu().numpy())

    pooled_np = np.concatenate(pooled_rows, axis=0)
    seq_np = np.concatenate(seq_rows, axis=0)
    return pooled_np, seq_np


def fit_control_to_sequence_regressor(
    pooled_controls: np.ndarray,
    seq_latents: np.ndarray,
    alpha: float = 1.0,
) -> Pipeline:
    """Fit a simple linear mapping from pooled controls to sequence latents."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )
    pipe.fit(pooled_controls, seq_latents)
    return pipe


def save_baseline_artifacts(
    output_dir: str,
    pooled_controls: np.ndarray,
    seq_latents: np.ndarray,
    regressor: Pipeline,
) -> dict[str, str]:
    """Save extracted arrays and regressor for later baseline decoding."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pooled_path = out / "pooled_controls.npy"
    seq_path = out / "seq_latents.npy"
    reg_path = out / "control_to_seq.pkl"

    np.save(pooled_path, pooled_controls)
    np.save(seq_path, seq_latents)
    with reg_path.open("wb") as fp:
        pickle.dump(regressor, fp)

    return {
        "pooled_controls": str(pooled_path),
        "seq_latents": str(seq_path),
        "control_to_seq": str(reg_path),
    }


def load_regressor(path: str) -> Pipeline:
    with open(path, "rb") as fp:
        return pickle.load(fp)


class PooledFeatureControlAdapter(torch.nn.Module):
    """Adapter that exposes encode_control/decode_control on top of a codec model."""

    def __init__(self, codec_model, regressor: Pipeline, device: torch.device):
        super().__init__()
        self.codec_model = codec_model
        self.regressor = regressor
        self.device = device

    def eval(self):
        self.codec_model.eval()
        return super().eval()

    def encode_control(self, x: torch.Tensor) -> torch.Tensor:
        features = self.codec_model.encode_features(x)
        return pool_codec_features(features)

    def decode_control(self, z_ctrl: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        z_np = z_ctrl.detach().cpu().numpy()
        seq_flat = self.regressor.predict(z_np)
        seq_flat_t = torch.from_numpy(seq_flat).float().to(self.device)
        bsz = seq_flat_t.shape[0]
        seq = seq_flat_t.view(
            bsz,
            self.codec_model.code_dim,
            self.codec_model.latent_frames,
        )
        return self.codec_model.decode_sequence(seq, target_len=target_len)

    def decode_sequence(self, z_seq: torch.Tensor, target_len: int | None = None) -> torch.Tensor:
        return self.codec_model.decode_sequence(z_seq, target_len=target_len)
