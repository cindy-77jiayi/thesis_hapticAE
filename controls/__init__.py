"""Compatibility exports for semantic control mapping."""

from .mapping import (
    attributes_to_pc_vector,
    load_control_schema,
    pc_vector_to_attributes,
    validate_attributes,
)

__all__ = [
    "attributes_to_pc_vector",
    "load_control_schema",
    "pc_vector_to_attributes",
    "validate_attributes",
]
