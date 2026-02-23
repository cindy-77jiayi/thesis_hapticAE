"""Step 1: Extract latent vectors (mu) from the full dataset using a trained VAE encoder."""

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
