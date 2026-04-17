import numpy as np

from src.data.preprocessing import augment_segment


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
