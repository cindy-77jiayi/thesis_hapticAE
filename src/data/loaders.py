"""Shared data loading and model construction helpers for scripts."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .preprocessing import collect_clean_wavs, estimate_global_rms
from .dataset import HapticWavDataset


def build_dataloaders(
    config: dict,
    data_dir: str,
    batch_size: int | None = None,
    full_dataset: bool = False,
) -> dict:
    """Build train/val DataLoaders (or a single full-dataset loader) from config.

    Args:
        config: Parsed YAML config dict.
        data_dir: Root directory of hapticgen-dataset.
        batch_size: Override config batch_size if provided.
        full_dataset: If True, return a single loader over all files
            (used for latent extraction). Train/val loaders are omitted.

    Returns:
        Dict with keys: 'wav_files', 'global_rms', and either
        'train_loader'/'val_loader' or 'all_loader'.
    """
    data_cfg = config["data"]
    bs = batch_size or config.get("training", {}).get("batch_size", 32)

    accepted_models = set(data_cfg.get("accepted_models", ["HapticGen"]))
    accepted_votes = set(data_cfg.get("accepted_votes", [1]))
    include_subdirs_cfg = data_cfg.get("include_subdirs")
    include_subdirs = set(include_subdirs_cfg) if include_subdirs_cfg else None

    wav_files = collect_clean_wavs(
        data_dir,
        accepted_models=accepted_models,
        accepted_votes=accepted_votes,
        include_subdirs=include_subdirs,
    )
    assert len(wav_files) > 0, f"No WAV files found in {data_dir}"

    N = len(wav_files)
    perm = np.random.permutation(N)
    split = int(data_cfg["train_split"] * N)
    train_files = [wav_files[i] for i in perm[:split]]

    global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])

    ds_kwargs = dict(
        T=data_cfg["T"],
        sr_expect=data_cfg["sr"],
        global_rms=global_rms,
        scale=data_cfg["scale"],
        use_minmax=data_cfg.get("use_minmax", False),
    )

    result = {"wav_files": wav_files, "global_rms": global_rms}

    if full_dataset:
        all_ds = HapticWavDataset(wav_files, **ds_kwargs)
        result["all_loader"] = DataLoader(
            all_ds, batch_size=bs, shuffle=False, drop_last=False,
        )
    else:
        val_files = [wav_files[i] for i in perm[split:]]
        train_ds = HapticWavDataset(train_files, **ds_kwargs)
        val_ds = HapticWavDataset(val_files, **ds_kwargs)
        result["train_loader"] = DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=True,
        )
        result["val_loader"] = DataLoader(
            val_ds, batch_size=bs, shuffle=False, drop_last=False,
        )

    return result


def build_model(config: dict, device: torch.device | None = None) -> nn.Module:
    """Instantiate ConvVAE or ConvAE from a config dict.

    Args:
        config: Parsed YAML config dict with 'model' and 'data' sections.
        device: If provided, moves model to this device.

    Returns:
        Instantiated model (eval mode is NOT set — caller decides).
    """
    from src.models.conv_vae import ConvVAE
    from src.models.conv_ae import ConvAE

    data_cfg = config["data"]
    model_cfg = config["model"]
    model_type = config.get("model_type", "vae")

    common = dict(
        T=data_cfg["T"],
        latent_dim=model_cfg["latent_dim"],
        channels=tuple(model_cfg["channels"]),
        first_kernel=model_cfg.get("first_kernel", 25),
        kernel_size=model_cfg.get("kernel_size", 9),
        activation=model_cfg.get("activation", "leaky_relu"),
        norm=model_cfg.get("norm", "group"),
    )

    if model_type == "vae":
        model = ConvVAE(
            **common,
            logvar_clip=tuple(model_cfg.get("logvar_clip", [-10, 10])),
        )
    else:
        model = ConvAE(**common)

    if device is not None:
        model = model.to(device)
    return model


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device | None = None,
) -> nn.Module:
    """Load a state_dict checkpoint into a model.

    Returns the model in eval mode on the given device.
    """
    map_loc = device or torch.device("cpu")
    state = torch.load(path, map_location=map_loc, weights_only=True)
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
