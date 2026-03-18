"""Deterministic mapping from normalized semantic attributes to PCA controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SCHEMA_PATH = Path(__file__).with_name("control_schema.json")


def load_control_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load the machine-readable control schema JSON."""
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def map_unit_to_pc(x: float, low: float, high: float) -> float:
    """Linearly map x in [0, 1] to [low, high]."""
    return low + float(x) * (high - low)


def validate_attributes(
    attrs: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
) -> dict[str, float]:
    """Validate and normalize semantic attributes.

    Args:
        attrs: Incoming attribute dict from LLM/rule-based output.
        schema: Optional loaded schema; defaults to control_schema.json.
        clip: If True, clips out-of-range numeric values to [0, 1].

    Returns:
        Dict with all expected attributes as floats in [0, 1].

    Raises:
        ValueError: Missing/unknown keys or non-numeric values.
    """
    schema = schema or load_control_schema()
    specs = schema["attributes"]
    expected = [s["name"] for s in specs]

    missing = [k for k in expected if k not in attrs]
    if missing:
        raise ValueError(f"Missing attributes: {missing}. Expected: {expected}")

    unknown = [k for k in attrs.keys() if k not in expected]
    if unknown:
        raise ValueError(f"Unknown attributes: {unknown}. Expected only: {expected}")

    validated: dict[str, float] = {}
    for spec in specs:
        name = spec["name"]
        value = attrs[name]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Attribute '{name}' must be numeric in [0,1], got {type(value).__name__}: {value}"
            )

        v = float(value)
        if clip:
            v = min(max(v, 0.0), 1.0)
        elif not (0.0 <= v <= 1.0):
            raise ValueError(f"Attribute '{name}' out of range [0,1]: {v}")

        validated[name] = v

    return validated


def attributes_to_pc_vector(
    attrs: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    clip: bool = True,
) -> list[float]:
    """Convert 4 normalized semantic attributes into an 8D PC vector.

    Mapping rule for MVP:
      - PC1..PC4 are linearly mapped from attributes.
      - PC5..PC8 are fixed to 0.
    """
    schema = schema or load_control_schema()
    validated = validate_attributes(attrs, schema=schema, clip=clip)

    pc_vector = list(schema.get("defaults", {}).get("pc_vector", [0.0] * 8))
    if len(pc_vector) != 8:
        raise ValueError(f"default pc_vector must be length 8, got {len(pc_vector)}")

    for spec in schema["attributes"]:
        name = spec["name"]
        idx = int(spec["pc_index"])
        low, high = spec["pc_range"]
        pc_vector[idx] = map_unit_to_pc(validated[name], float(low), float(high))

    # Force the tail dimensions to zero for MVP invariance.
    for i in range(4, 8):
        pc_vector[i] = 0.0

    return [float(v) for v in pc_vector]
