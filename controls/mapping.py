"""Backward-compatible wrappers around the canonical semantic control mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.semantic.mapping import (
    normalize_semantic_controls,
    pca_to_semantic,
    semantic_to_pca,
)
from src.semantic.pc_semantics import load_semantic_schema


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "semantic_control_schema.json"


def load_control_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load the canonical semantic control schema."""
    return load_semantic_schema(schema_path or DEFAULT_SCHEMA_PATH)


def validate_attributes(
    attrs: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
) -> dict[str, float]:
    """Compatibility wrapper for semantic input validation.

    The returned dict always uses canonical semantic keys:
    `frequency`, `intensity`, `envelope_modulation`, `temporal_grouping`, `sharpness`.
    """
    return normalize_semantic_controls(attrs, schema=schema, clip=clip, require_all=True)


def attributes_to_pc_vector(
    attrs: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
) -> list[float]:
    """Compatibility wrapper that maps semantic controls into an 8D PCA vector."""
    vector = semantic_to_pca(attrs, schema=schema, clip=clip, require_all=True)
    return [float(v) for v in vector.tolist()]


def pc_vector_to_attributes(
    pc_vector: list[float],
    schema: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Map an 8D PCA vector back to canonical semantic attributes."""
    return pca_to_semantic(pc_vector, schema=schema)
