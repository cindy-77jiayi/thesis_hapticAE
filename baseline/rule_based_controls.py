"""Rule-based baseline for semantic haptic attributes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_ATTRS: dict[str, float] = {
    "energy_roughness": 0.5,
    "temporal_irregularity": 0.5,
    "modulation_texture": 0.5,
    "decay_envelope": 0.5,
}


# Baseline presets for quick MVP comparison vs. LLM predictions.
ACTION_TYPE_PRESETS: dict[str, dict[str, float]] = {
    "success_confirmation": {
        "energy_roughness": 0.7,
        "temporal_irregularity": 0.1,
        "modulation_texture": 0.4,
        "decay_envelope": 0.6,
    },
    "error_alert": {
        "energy_roughness": 0.8,
        "temporal_irregularity": 0.8,
        "modulation_texture": 0.7,
        "decay_envelope": 0.2,
    },
    "toggle_on": {
        "energy_roughness": 0.5,
        "temporal_irregularity": 0.1,
        "modulation_texture": 0.3,
        "decay_envelope": 0.3,
    },
    "toggle_off": {
        "energy_roughness": 0.4,
        "temporal_irregularity": 0.1,
        "modulation_texture": 0.2,
        "decay_envelope": 0.25,
    },
    "slider_drag": {
        "energy_roughness": 0.45,
        "temporal_irregularity": 0.35,
        "modulation_texture": 0.65,
        "decay_envelope": 0.35,
    },
    "success_state": {
        "energy_roughness": 0.6,
        "temporal_irregularity": 0.15,
        "modulation_texture": 0.35,
        "decay_envelope": 0.55,
    },
    "warning_alert": {
        "energy_roughness": 0.65,
        "temporal_irregularity": 0.5,
        "modulation_texture": 0.55,
        "decay_envelope": 0.35,
    },
    "long_press": {
        "energy_roughness": 0.55,
        "temporal_irregularity": 0.2,
        "modulation_texture": 0.45,
        "decay_envelope": 0.75,
    },
    "selection_change": {
        "energy_roughness": 0.4,
        "temporal_irregularity": 0.2,
        "modulation_texture": 0.3,
        "decay_envelope": 0.3,
    },
    "cancel_action": {
        "energy_roughness": 0.35,
        "temporal_irregularity": 0.15,
        "modulation_texture": 0.2,
        "decay_envelope": 0.25,
    },
}


def get_rule_based_attributes(action_type: str | None, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    """Return baseline attributes for an action type.

    Falls back to neutral defaults when action_type is missing/unknown.
    """
    attrs = deepcopy(DEFAULT_ATTRS)
    if action_type and action_type in ACTION_TYPE_PRESETS:
        attrs.update(ACTION_TYPE_PRESETS[action_type])
        return attrs

    # Optional lightweight fallback hinting from action_name when type is absent.
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
