"""Deterministic mapping between semantic control space and PCA control space."""

from __future__ import annotations

from typing import Any

import numpy as np

from .pc_semantics import CANONICAL_SEMANTIC_ORDER, get_control_spec, load_semantic_schema


def _clip_unit(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _build_alias_lookup(schema: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for spec in schema["semantic_controls"]:
        canonical = spec["canonical_name"]
        lookup[canonical] = canonical
        for alias in spec.get("aliases", []):
            lookup[str(alias)] = canonical
    return lookup


def semantic_intensity_to_pc2(value: float, low: float, high: float) -> float:
    """Map semantic intensity to the inverted PC2 axis."""
    value = _clip_unit(value)
    return float(high - value * (high - low))


def pc2_to_semantic_intensity(pc2_value: float, low: float, high: float) -> float:
    """Map the inverted PC2 axis back to semantic intensity."""
    if high == low:
        return 0.5
    return _clip_unit((float(high) - float(pc2_value)) / (float(high) - float(low)))


def normalize_semantic_controls(
    raw_controls: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
    require_all: bool = True,
) -> dict[str, float]:
    """Resolve aliases and return a normalized canonical semantic control dict."""
    schema = schema or load_semantic_schema()
    alias_lookup = _build_alias_lookup(schema)
    defaults = {spec["canonical_name"]: float(spec.get("default", 0.5)) for spec in schema["semantic_controls"]}
    normalized = dict(defaults)

    seen: dict[str, str] = {}
    unknown = []
    for key, value in raw_controls.items():
        canonical = alias_lookup.get(str(key))
        if canonical is None:
            unknown.append(str(key))
            continue
        if canonical in seen and seen[canonical] != key:
            raise ValueError(f"Multiple aliases provided for '{canonical}': {seen[canonical]!r} and {key!r}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Semantic control '{key}' must be numeric, got {type(value).__name__}: {value!r}")
        numeric = float(value)
        if clip:
            numeric = _clip_unit(numeric)
        elif not (0.0 <= numeric <= 1.0):
            raise ValueError(f"Semantic control '{key}' out of range [0,1]: {numeric}")
        normalized[canonical] = numeric
        seen[canonical] = str(key)

    if unknown:
        allowed = sorted(alias_lookup)
        raise ValueError(f"Unknown semantic controls: {unknown}. Allowed keys/aliases: {allowed}")

    if require_all:
        missing = [name for name in CANONICAL_SEMANTIC_ORDER if name not in seen]
        if missing:
            provided = sorted(raw_controls.keys())
            raise ValueError(
                f"Missing semantic controls: {missing}. Provided keys: {provided}. "
                f"Expected canonical keys: {CANONICAL_SEMANTIC_ORDER}"
            )

    return {name: float(normalized[name]) for name in CANONICAL_SEMANTIC_ORDER}


def semantic_to_pca(
    raw_or_normalized_semantic_dict: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
    require_all: bool = True,
    vector_length: int = 8,
) -> np.ndarray:
    """Map semantic controls into an 8D PCA control vector."""
    schema = schema or load_semantic_schema()
    normalized = normalize_semantic_controls(
        raw_or_normalized_semantic_dict,
        schema=schema,
        clip=clip,
        require_all=require_all,
    )
    pc_vector = np.zeros(vector_length, dtype=np.float32)

    for canonical_name in CANONICAL_SEMANTIC_ORDER:
        spec = get_control_spec(canonical_name, schema=schema)
        low, high = [float(v) for v in spec["pc_range"]]
        semantic_value = normalized[canonical_name]
        if spec["transform"]["type"] == "inverted_identity":
            pc_value = semantic_intensity_to_pc2(semantic_value, low, high)
        else:
            pc_value = low + semantic_value * (high - low)
        pc_vector[int(spec["pc_index"])] = float(pc_value)

    return pc_vector


def pca_to_semantic(
    pca_vector: list[float] | np.ndarray,
    schema: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Project an 8D PCA vector back into canonical semantic space for PC1-PC5."""
    schema = schema or load_semantic_schema()
    vector = np.asarray(pca_vector, dtype=np.float32).reshape(-1)
    if vector.size < 5:
        raise ValueError(f"PCA vector must have at least 5 dimensions, got {vector.size}")

    result: dict[str, float] = {}
    for canonical_name in CANONICAL_SEMANTIC_ORDER:
        spec = get_control_spec(canonical_name, schema=schema)
        low, high = [float(v) for v in spec["pc_range"]]
        pc_value = float(vector[int(spec["pc_index"])])
        if spec["transform"]["type"] == "inverted_identity":
            semantic_value = pc2_to_semantic_intensity(pc_value, low, high)
        else:
            if high == low:
                semantic_value = 0.5
            else:
                semantic_value = _clip_unit((pc_value - low) / (high - low))
        result[canonical_name] = float(semantic_value)
    return result


def generate_from_semantic(
    semantic_dict: dict[str, Any],
    pipe: Any,
    model: Any,
    device: Any,
    *,
    target_len: int,
    schema: dict[str, Any] | None = None,
):
    """Generate a waveform from semantic controls using an existing PCA pipe and decoder."""
    import torch

    from src.pipelines.pca_control import control_to_latent

    pc_vector = semantic_to_pca(semantic_dict, schema=schema)
    latent = control_to_latent(pipe, pc_vector.astype(np.float32))
    latent_t = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    signal = model.decode(latent_t, target_len=target_len).squeeze().detach().cpu().numpy()
    return signal, pc_vector
