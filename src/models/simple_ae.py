"""Minimal fully-connected autoencoder for sanity checks."""

import torch
import torch.nn as nn


class SimpleAE(nn.Module):
    """Fully-connected autoencoder — no convolutions.

    Used as a baseline / sanity-check model.
    """

    def __init__(self, T: int = 1000, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(T, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, T), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T) -> flatten
        x_flat = x.squeeze(1)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z).unsqueeze(1)
        return x_hat, z
