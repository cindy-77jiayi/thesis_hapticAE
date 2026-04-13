from __future__ import annotations

import numpy as np

from src.semantic.mapping import (
    normalize_semantic_controls,
    pca_to_semantic,
    semantic_to_pca,
)
from src.semantic.pc_semantics import load_semantic_schema


def test_semantic_to_pca_maps_frequency_to_pc1_only():
    schema = load_semantic_schema()
    controls = {
        "frequency": 1.0,
        "intensity": 0.5,
        "envelope_modulation": 0.5,
        "temporal_grouping": 0.5,
        "sharpness": 0.5,
    }
    vector = semantic_to_pca(controls, schema=schema)
    assert vector.shape == (8,)
    assert np.isclose(vector[0], schema["semantic_controls"][0]["pc_range"][1])


def test_intensity_is_inverted_on_pc2():
    schema = load_semantic_schema()
    base = {
        "frequency": 0.5,
        "intensity": 0.0,
        "envelope_modulation": 0.5,
        "temporal_grouping": 0.5,
        "sharpness": 0.5,
    }
    high = dict(base, intensity=1.0)
    low_vec = semantic_to_pca(base, schema=schema)
    high_vec = semantic_to_pca(high, schema=schema)
    assert high_vec[1] < low_vec[1]


def test_aliases_are_normalized():
    schema = load_semantic_schema()
    normalized = normalize_semantic_controls(
        {
            "pc1": 0.2,
            "pc2": 0.8,
            "pc3": 0.4,
            "pc4": 0.6,
            "pc5": 0.7,
        },
        schema=schema,
    )
    assert normalized == {
        "frequency": 0.2,
        "intensity": 0.8,
        "envelope_modulation": 0.4,
        "temporal_grouping": 0.6,
        "sharpness": 0.7,
    }


def test_round_trip_preserves_semantic_direction():
    schema = load_semantic_schema()
    controls = {
        "frequency": 0.65,
        "intensity": 0.8,
        "envelope_modulation": 0.3,
        "temporal_grouping": 0.4,
        "sharpness": 0.9,
    }
    vector = semantic_to_pca(controls, schema=schema)
    recovered = pca_to_semantic(vector, schema=schema)
    for key, expected in controls.items():
        assert np.isclose(recovered[key], expected, atol=1e-5)


def test_tail_dimensions_stay_zero():
    schema = load_semantic_schema()
    vector = semantic_to_pca(
        {
            "frequency": 0.5,
            "intensity": 0.5,
            "envelope_modulation": 0.5,
            "temporal_grouping": 0.5,
            "sharpness": 0.5,
        },
        schema=schema,
    )
    assert np.allclose(vector[5:], 0.0)
