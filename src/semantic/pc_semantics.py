"""Helpers for the canonical semantic control schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "semantic_control_schema.json"
CANONICAL_SEMANTIC_ORDER = [
    "frequency",
    "intensity",
    "envelope_modulation",
    "temporal_grouping",
    "sharpness",
]


def load_semantic_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    """Load the canonical semantic control schema."""
    path = Path(schema_path) if schema_path else SCHEMA_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_control_spec(canonical_name: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one canonical semantic control spec by name."""
    schema = schema or load_semantic_schema()
    for spec in schema["semantic_controls"]:
        if spec["canonical_name"] == canonical_name:
            return spec
    raise KeyError(f"Unknown canonical semantic control: {canonical_name}")


def get_unresolved_pc_specs(schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return unresolved PC placeholder specs."""
    schema = schema or load_semantic_schema()
    return list(schema.get("unresolved_pcs", []))


def build_semantic_control_table(schema: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build a human-readable control table from the schema."""
    schema = schema or load_semantic_schema()
    rows: list[dict[str, Any]] = []
    for spec in schema["semantic_controls"]:
        rows.append(
            {
                "control_id": spec["pc_name"],
                "canonical_name": spec["canonical_name"],
                "pc_index": spec["pc_index"],
                "control_name": spec["control_name"],
                "description": spec["description"],
                "low_end_interpretation": spec["low_end_interpretation"],
                "high_end_interpretation": spec["high_end_interpretation"],
                "semantic_range": spec.get("semantic_range", [0.0, 1.0]),
                "pc_range": spec.get("pc_range", []),
                "range_source": spec.get("range_source", ""),
                "semantic_space_direction": spec["semantic_space_direction"],
                "pca_space_direction": spec["pca_space_direction"],
                "inversion": bool(spec["inversion"]),
                "status": spec["status"],
                "notes": spec.get("notes", ""),
            }
        )
    for spec in schema.get("unresolved_pcs", []):
        rows.append(
            {
                "control_id": spec["pc_name"],
                "canonical_name": spec["canonical_name"],
                "pc_index": spec["pc_index"],
                "control_name": spec["control_name"],
                "description": spec["description"],
                "low_end_interpretation": "",
                "high_end_interpretation": "",
                "semantic_range": [],
                "pc_range": [],
                "range_source": "",
                "semantic_space_direction": "",
                "pca_space_direction": "",
                "inversion": False,
                "status": spec["status"],
                "notes": spec.get("notes", ""),
            }
        )
    return rows
