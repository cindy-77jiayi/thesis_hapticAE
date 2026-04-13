"""Rule-based baseline for canonical semantic haptic controls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_ATTRS: dict[str, float] = {
    "frequency": 0.5,
    "intensity": 0.5,
    "envelope_modulation": 0.5,
    "temporal_grouping": 0.5,
    "sharpness": 0.5,
}


# Coarse fallback presets for compatibility when no LLM semantic output is available.
ACTION_TYPE_PRESETS: dict[str, dict[str, float]] = {
    "success_confirmation": {
        "frequency": 0.55,
        "intensity": 0.70,
        "envelope_modulation": 0.35,
        "temporal_grouping": 0.20,
        "sharpness": 0.45,
    },
    "error_alert": {
        "frequency": 0.75,
        "intensity": 0.85,
        "envelope_modulation": 0.70,
        "temporal_grouping": 0.80,
        "sharpness": 0.80,
    },
    "toggle_on": {
        "frequency": 0.60,
        "intensity": 0.50,
        "envelope_modulation": 0.20,
        "temporal_grouping": 0.15,
        "sharpness": 0.55,
    },
    "toggle_off": {
        "frequency": 0.45,
        "intensity": 0.40,
        "envelope_modulation": 0.15,
        "temporal_grouping": 0.15,
        "sharpness": 0.45,
    },
    "slider_drag": {
        "frequency": 0.65,
        "intensity": 0.45,
        "envelope_modulation": 0.55,
        "temporal_grouping": 0.40,
        "sharpness": 0.40,
    },
    "success_state": {
        "frequency": 0.50,
        "intensity": 0.60,
        "envelope_modulation": 0.25,
        "temporal_grouping": 0.20,
        "sharpness": 0.35,
    },
    "warning_alert": {
        "frequency": 0.65,
        "intensity": 0.70,
        "envelope_modulation": 0.50,
        "temporal_grouping": 0.55,
        "sharpness": 0.65,
    },
    "long_press": {
        "frequency": 0.35,
        "intensity": 0.55,
        "envelope_modulation": 0.35,
        "temporal_grouping": 0.20,
        "sharpness": 0.25,
    },
    "selection_change": {
        "frequency": 0.55,
        "intensity": 0.40,
        "envelope_modulation": 0.20,
        "temporal_grouping": 0.25,
        "sharpness": 0.45,
    },
    "cancel_action": {
        "frequency": 0.40,
        "intensity": 0.35,
        "envelope_modulation": 0.20,
        "temporal_grouping": 0.20,
        "sharpness": 0.40,
    },
}


def get_rule_based_attributes(action_type: str | None, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Return canonical semantic controls for an action type.

    This is only a heuristic fallback when no structured semantic output is available.
    """
    attrs = deepcopy(DEFAULT_ATTRS)
    if action_type and action_type in ACTION_TYPE_PRESETS:
        attrs.update(ACTION_TYPE_PRESETS[action_type])
        return attrs

    action_name = (metadata or {}).get("action_name", "")
    if isinstance(action_name, str):
        name = action_name.lower()
        if "error" in name:
            attrs.update(ACTION_TYPE_PRESETS["error_alert"])
        elif "success" in name or "confirm" in name:
            attrs.update(ACTION_TYPE_PRESETS["success_confirmation"])
        elif "toggle" in name and "on" in name:
            attrs.update(ACTION_TYPE_PRESETS["toggle_on"])
        elif "toggle" in name and "off" in name:
            attrs.update(ACTION_TYPE_PRESETS["toggle_off"])

    return attrs
