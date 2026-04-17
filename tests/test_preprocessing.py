import numpy as np
from unittest.mock import patch

from src.data.preprocessing import augment_segment, load_segment_energy, stable_segment_seed


def test_augment_segment_applies_dropout_with_fixed_rng():
    seg = np.ones(8, dtype=np.float32)
    rng = np.random.default_rng(0)

    augmented = augment_segment(
        seg,
        gain_range=(1.0, 1.0),
        noise_std=0.0,
        shift_max=0,
        dropout_prob=1.0,
        dropout_width=3,
        rng=rng,
    )

    assert augmented.shape == seg.shape
    assert np.count_nonzero(augmented == 0.0) == 3
    assert np.count_nonzero(augmented == 1.0) == 5


def test_augment_segment_respects_identity_settings():
    seg = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    rng = np.random.default_rng(123)

    augmented = augment_segment(
        seg,
        gain_range=(1.0, 1.0),
        noise_std=0.0,
        shift_max=0,
        dropout_prob=0.0,
        dropout_width=0,
        rng=rng,
    )

    assert np.allclose(augmented, seg)


def test_load_segment_energy_is_repeatable_with_stable_rng_seed():
    signal = np.linspace(-1.0, 1.0, 64, dtype=np.float32)

    def fake_load_audio(path, target_sr=8000, target_channels=1):
        return signal[None, :], target_sr

    with patch("src.data.preprocessing.load_audio", side_effect=fake_load_audio):
        first = load_segment_energy(
            "demo.wav",
            T=16,
            sr_expect=8000,
            global_rms=1.0,
            scale=1.0,
            clip_range=(-5.0, 5.0),
            tries=8,
            min_energy=0.0,
            max_resample=3,
            search_window_seconds=0.004,
            top_k=4,
            random_segment_prob=0.5,
            augment=False,
            rng=np.random.default_rng(stable_segment_seed("demo.wav", 0)),
        )
        second = load_segment_energy(
            "demo.wav",
            T=16,
            sr_expect=8000,
            global_rms=1.0,
            scale=1.0,
            clip_range=(-5.0, 5.0),
            tries=8,
            min_energy=0.0,
            max_resample=3,
            search_window_seconds=0.004,
            top_k=4,
            random_segment_prob=0.5,
            augment=False,
            rng=np.random.default_rng(stable_segment_seed("demo.wav", 0)),
        )

    assert np.allclose(first, second)
