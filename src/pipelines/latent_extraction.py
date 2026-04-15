"""Step 1: Extract latent vectors (mu) from the full dataset using a trained VAE encoder."""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_latent_vectors(
    model,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Encode the entire dataset and return the mu vectors.

    Uses the deterministic mu (not sampled z) for stable, repeatable
    latent representations suitable for downstream analysis like PCA.

    Returns:
        Z: np.ndarray of shape (N, latent_dim) containing all mu vectors.
    """
    model.eval()
    all_mu = []

    with torch.no_grad():
        for x in tqdm(loader, desc="Extracting latents", leave=False):
            x = x.to(device)
            _, mu, _ = model.encode(x)
            all_mu.append(mu.cpu().numpy())

    Z = np.concatenate(all_mu, axis=0)

    assert Z.ndim == 2, f"Expected 2D array, got shape {Z.shape}"
    assert np.all(np.isfinite(Z)), f"Z contains {np.sum(~np.isfinite(Z))} NaN/Inf values"

    print(f"Extracted latent vectors: shape={Z.shape}, "
          f"mean={Z.mean():.4f}, std={Z.std():.4f}, "
          f"min={Z.min():.4f}, max={Z.max():.4f}")

    return Z


def load_or_fit_pca(
    model,
    config: dict,
    data_dir: str,
    device: torch.device,
    pca_dir: str | None = None,
    n_components: int = 8,
    save_dir: str | None = None,
):
    """Load an existing PCA pipeline or extract latents and fit a new one.

    Args:
        model: Trained VAE (already on device, eval mode).
        config: Parsed YAML config dict.
        data_dir: Root of dataset audio files.
        device: torch device.
        pca_dir: If provided and contains pca_pipe.pkl, load from there.
        n_components: Number of PCA components.
        save_dir: Where to save newly fitted PCA artifacts.

    Returns:
        (pipe, Z_pca) — sklearn Pipeline and projected scores.
    """
    from src.pipelines.pca_control import fit_pca_pipeline

    if pca_dir and os.path.exists(os.path.join(pca_dir, "pca_pipe.pkl")):
        print(f"📦 Loading existing PCA from {pca_dir}")
        with open(os.path.join(pca_dir, "pca_pipe.pkl"), "rb") as f:
            pipe = pickle.load(f)
        Z_pca = np.load(os.path.join(pca_dir, "Z_pca.npy"))
        return pipe, Z_pca

    from src.data.loaders import build_dataloaders

    data = build_dataloaders(config, data_dir, batch_size=64, full_dataset=True)
    print(f"   Dataset size: {len(data['audio_files'])}")

    Z = extract_latent_vectors(model, data["all_loader"], device)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "Z.npy"), Z)

    pipe, Z_pca = fit_pca_pipeline(Z, n_components=n_components, save_dir=save_dir)
    return pipe, Z_pca
