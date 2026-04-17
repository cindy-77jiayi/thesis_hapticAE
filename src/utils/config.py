"""YAML config loading with defaults."""

from __future__ import annotations

import copy
import os

import yaml


_DEFAULTS = {
    "seed": 42,
    "model_type": "vae",
    "run_name": None,

    "data": {
        "sr": 8000,
        "T": 4000,
        "scale": 0.25,
        "use_minmax": False,
        "train_split": 0.8,
        "clip_range": [-3.0, 3.0],
        "segment_tries": 30,
        "segment_top_k": 4,
        "max_resample": 5,
        "min_energy": 5e-4,
        "random_segment_prob": 0.0,
        "search_window_seconds": None,
        "augmentation": {
            "enabled": False,
            "gain_range": [0.9, 1.1],
            "noise_std": 0.0,
            "shift_max": 0,
            "dropout_prob": 0.0,
            "dropout_width": 0,
        },
    },

    "model": {
        "latent_dim": 64,
        "channels": [32, 64, 128, 128],
        "first_kernel": 25,
        "kernel_size": 9,
        "activation": "leaky_relu",
        "norm": "group",
        "logvar_clip": [-10.0, 10.0],
    },

    "training": {
        "batch_size": 32,
        "epochs": 100,
        "patience": 15,
        "min_delta": 1e-4,
        "early_stop_start": 10,
        "grad_clip": 1.0,
        "print_every": 10,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
    },

    "optimizer": {
        "lr": 2e-4,
        "weight_decay": 1e-5,
    },

    "scheduler": {
        "factor": 0.5,
        "patience": 15,
    },

    "ema": {
        "use": False,
        "decay": 0.999,
    },

    "checkpoint": {
        "save_last": True,
        "save_best": True,
        "save_every": 0,
        "keep_last": 0,
    },

    "validation": {
        "sample_every": 0,
        "n_samples": 4,
        "deterministic_vae": True,
    },

    "loss": {
        "l1_weight": 0.2,
        "spectral_weight": 0.15,
        "amplitude_weight": 0.5,
        "fft_weight": 0.0,
        "clamp_range": 3.0,
    },

    "kl": {
        "free_bits": 0.1,
        "beta_max": 0.0001,
        "n_cycles": 4,
        "ratio": 0.5,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _parse_override_value(raw: str):
    """Parse a CLI override value using YAML semantics."""
    return yaml.safe_load(raw)


def apply_overrides(config: dict, overrides: list[str] | None = None) -> dict:
    """Apply dotlist overrides such as ``training.epochs=20``."""
    if not overrides:
        return config

    updated = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value syntax.")
        key, raw_value = item.split("=", 1)
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Invalid override key in '{item}'.")

        cursor = updated
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = _parse_override_value(raw_value)
    return updated


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_config_recursive(path: str, visited: set[str]) -> dict:
    resolved = os.path.abspath(path)
    if resolved in visited:
        raise ValueError(f"Config inheritance cycle detected at {resolved}")
    next_visited = set(visited)
    next_visited.add(resolved)

    user_cfg = _load_yaml(resolved)
    base_entry = user_cfg.pop("base_config", None)
    if base_entry is None:
        return user_cfg

    base_paths = base_entry if isinstance(base_entry, list) else [base_entry]
    merged: dict = {}
    for base_path in base_paths:
        base_resolved = base_path
        if not os.path.isabs(base_resolved):
            base_resolved = os.path.join(os.path.dirname(resolved), base_resolved)
        merged = _deep_merge(merged, _load_config_recursive(base_resolved, next_visited))
    return _deep_merge(merged, user_cfg)


def dump_config(path: str, config: dict) -> None:
    """Write a resolved config to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config file, resolve inheritance, and merge with defaults."""
    resolved = _load_config_recursive(path, visited=set())
    config = _deep_merge(copy.deepcopy(_DEFAULTS), resolved)
    return apply_overrides(config, overrides)
