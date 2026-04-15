"""YAML config loading with defaults."""

import yaml


_DEFAULTS = {
    "seed": 42,
    "model_type": "vae",
    "run_name": None,

    "data": {
        "sr": 8000,
        "T": 80000,
        "scale": 1.0,
        "extensions": [".wav", ".flac"],
        "segment_mode": "hapticgen",
        "normalize_mode": "none",
        "min_segment_ratio": 1.0,
        "clip_range": None,
        "use_minmax": False,
        "train_split": 0.8,
        "analysis_batch_size": 4,
        "train_random_seek": True,
        "train_sample_with_replacement": False,
        "val_random_seek": False,
        "val_sample_with_replacement": False,
        "train_file_list": None,
        "val_file_list": None,
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
        "batch_size": 4,
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
        "stft_loss_weight": 0.15,
        "amplitude_weight": 0.5,
        "clamp_range": 3.0,
        "recon_time_weight": 1.0,
        "stft_scales": [128, 256, 512, 1024],
        "stft_hop_lengths": None,
        "stft_win_lengths": None,
        "stft_scale_weights": None,
        "stft_linear_weight": 1.0,
        "stft_log_weight": 1.0,
        "stft_eps": 1e-7,
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
