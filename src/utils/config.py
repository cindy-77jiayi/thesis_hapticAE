"""YAML config loading with defaults."""

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
    },

    "optimizer": {
        "lr": 2e-4,
        "weight_decay": 1e-5,
    },

    "scheduler": {
        "factor": 0.5,
        "patience": 15,
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


def load_config(path: str) -> dict:
    """Load a YAML config file and merge with defaults."""
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(_DEFAULTS, user_cfg)
