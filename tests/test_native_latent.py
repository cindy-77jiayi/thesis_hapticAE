from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.pipelines.native_latent import compute_latent_ranges, summarize_latent_dimensions


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from src.pipelines.native_latent import sweep_latent_axis


def test_summarize_latent_dimensions_reports_axis_stats():
    Z = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_latent_dimensions(Z, quantiles=(0.0, 0.5, 1.0))

    assert len(summary) == 2
    assert summary[0]["axis_label"] == "z1"
    assert np.isclose(summary[0]["mean"], 2.0)
    assert np.isclose(summary[0]["q00"], 0.0)
    assert np.isclose(summary[0]["q50"], 2.0)
    assert np.isclose(summary[0]["q100"], 4.0)
    assert summary[1]["axis_label"] == "z2"
    assert np.isclose(summary[1]["std"], np.std(Z[:, 1]))


def test_compute_latent_ranges_uses_empirical_quantiles():
    Z = np.array(
        [
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
            [3.0, 13.0],
            [4.0, 14.0],
        ],
        dtype=np.float32,
    )

    ranges = compute_latent_ranges(Z, low_q=0.2, high_q=0.8)

    assert [item["axis_label"] for item in ranges] == ["z1", "z2"]
    assert np.isclose(ranges[0]["low"], np.quantile(Z[:, 0], 0.2))
    assert np.isclose(ranges[0]["high"], np.quantile(Z[:, 0], 0.8))
    assert np.isclose(ranges[1]["mean"], np.mean(Z[:, 1]))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the current environment")
def test_sweep_latent_axis_uses_zero_reference_when_requested():
    class DummyModel:
        def eval(self):
            return self

        def decode(self, z, target_len):
            values = z.sum(dim=1, keepdim=True).unsqueeze(-1)
            return values.repeat(1, 1, target_len)

    device = torch.device("cpu")
    result = sweep_latent_axis(
        DummyModel(),
        device,
        axis=1,
        sweep_range=(-1.0, 1.0),
        n_steps=3,
        T=4,
        latent_dim=3,
    )

    assert result["axis_label"] == "z2"
    assert result["latents"].shape == (3, 3)
    assert result["signals"].shape == (3, 4)
    assert np.allclose(result["latents"][:, 0], 0.0)
    assert np.allclose(result["latents"][:, 2], 0.0)
    assert np.allclose(result["latents"][:, 1], [-1.0, 0.0, 1.0])
    assert np.allclose(result["signals"][:, 0], [-1.0, 0.0, 1.0])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed in the current environment")
def test_sweep_latent_axis_preserves_reference_on_unswept_dims():
    class DummyModel:
        def eval(self):
            return self

        def decode(self, z, target_len):
            values = z.sum(dim=1, keepdim=True).unsqueeze(-1)
            return values.repeat(1, 1, target_len)

    device = torch.device("cpu")
    reference = np.array([0.25, -0.5, 1.5], dtype=np.float32)

    result = sweep_latent_axis(
        DummyModel(),
        device,
        axis=0,
        sweep_range=(0.0, 0.5),
        n_steps=2,
        T=2,
        reference=reference,
    )

    assert np.allclose(result["latents"][:, 1], -0.5)
    assert np.allclose(result["latents"][:, 2], 1.5)
    assert np.allclose(result["latents"][:, 0], [0.0, 0.5])
    assert np.allclose(result["signals"][:, 0], [1.0, 1.5])
