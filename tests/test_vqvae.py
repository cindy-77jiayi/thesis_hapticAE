import importlib.util

import pytest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.data.loaders import build_model
    from src.models.conv_vqvae import ConvVQVAE


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the current environment")
def test_conv_vqvae_forward_shapes():
    model = ConvVQVAE(
        T=128,
        channels=(8, 16),
        first_kernel=7,
        kernel_size=5,
        embedding_dim=4,
        codebook_size=8,
    )
    x = torch.zeros(2, 1, 128)

    x_hat, z_q, codes, vq_loss, perplexity = model(x)

    assert x_hat.shape == x.shape
    assert z_q.shape[:2] == (2, 4)
    assert codes.shape == (2, z_q.shape[-1])
    assert vq_loss.ndim == 0
    assert perplexity.ndim == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the current environment")
def test_build_model_supports_vqvae_without_global_latent_dim():
    config = {
        "model_type": "vqvae",
        "data": {"T": 128},
        "model": {
            "channels": [8, 16],
            "first_kernel": 7,
            "kernel_size": 5,
            "activation": "leaky_relu",
            "norm": "group",
        },
        "vq": {
            "embedding_dim": 4,
            "codebook_size": 8,
            "commitment_cost": 0.25,
        },
    }

    model = build_model(config)

    assert isinstance(model, ConvVQVAE)
