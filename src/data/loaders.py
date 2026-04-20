"""Shared data loading and model construction helpers for scripts."""

import json
import os

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
    split_manifest_path: str | None = None,
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
    train_cfg = config.get("training", {})
    bs = batch_size or train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)
    pin_memory = bool(train_cfg.get("pin_memory", False))
    persistent_workers = bool(train_cfg.get("persistent_workers", False) and num_workers > 0)
    prefetch_factor = train_cfg.get("prefetch_factor", None)

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

    ds_kwargs = dict(
        T=data_cfg["T"],
        sr_expect=data_cfg["sr"],
        scale=data_cfg["scale"],
        use_minmax=data_cfg.get("use_minmax", False),
        clip_range=tuple(data_cfg.get("clip_range", [-3.0, 3.0])),
        segment_tries=int(data_cfg.get("segment_tries", 30)),
        min_energy=float(data_cfg.get("min_energy", 5e-4)),
        max_resample=int(data_cfg.get("max_resample", 5)),
        search_window_seconds=data_cfg.get("search_window_seconds", None),
        segment_top_k=int(data_cfg.get("segment_top_k", 4)),
        random_segment_prob=float(data_cfg.get("random_segment_prob", 0.0)),
    )
    aug_cfg = data_cfg.get("augmentation", {})
    use_augmentation = bool(aug_cfg.get("enabled", False))

    result = {"wav_files": wav_files}

    if full_dataset:
        global_rms = estimate_global_rms(wav_files, n=200, sr_expect=data_cfg["sr"])
        result["global_rms"] = global_rms
        all_ds = HapticWavDataset(
            wav_files,
            global_rms=global_rms,
            augment=False,
            augmentation_config={},
            deterministic=True,
            **ds_kwargs,
        )
        loader_kwargs = {
            "batch_size": bs,
            "shuffle": False,
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": persistent_workers,
        }
        if prefetch_factor is not None and num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        result["all_loader"] = DataLoader(all_ds, **loader_kwargs)
        return result

    train_files: list[str]
    val_files: list[str]
    global_rms: float
    if split_manifest_path and os.path.exists(split_manifest_path):
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_files = manifest["train_files"]
        val_files = manifest["val_files"]
        global_rms = float(manifest["global_rms"])
    else:
        N = len(wav_files)
        perm = np.random.permutation(N)
        split = int(data_cfg["train_split"] * N)
        train_files = [wav_files[i] for i in perm[:split]]
        val_files = [wav_files[i] for i in perm[split:]]
        global_rms = estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])
        if split_manifest_path:
            manifest = {
                "data_dir": os.path.abspath(data_dir),
                "global_rms": global_rms,
                "train_files": train_files,
                "val_files": val_files,
            }
            os.makedirs(os.path.dirname(split_manifest_path), exist_ok=True)
            with open(split_manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

    result["global_rms"] = global_rms
    result["train_files"] = train_files
    result["val_files"] = val_files

    train_ds = HapticWavDataset(
        train_files,
        global_rms=global_rms,
        augment=use_augmentation,
        augmentation_config=aug_cfg,
        deterministic=False,
        **ds_kwargs,
    )
    val_ds_kwargs = dict(ds_kwargs)
    val_ds_kwargs["segment_top_k"] = 1
    val_ds_kwargs["random_segment_prob"] = 0.0
    val_ds_kwargs["search_window_seconds"] = None
    val_ds = HapticWavDataset(
        val_files,
        global_rms=global_rms,
        augment=False,
        augmentation_config={},
        deterministic=True,
        **val_ds_kwargs,
    )
    loader_common = {
        "batch_size": bs,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_common["prefetch_factor"] = prefetch_factor

    result["train_loader"] = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )
    result["val_loader"] = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    return result


def build_model(config: dict, device: torch.device | None = None) -> nn.Module:
    """Instantiate ConvVAE, ConvAE, or ConvVQVAE from a config dict.

    Args:
        config: Parsed YAML config dict with 'model' and 'data' sections.
        device: If provided, moves model to this device.

    Returns:
        Instantiated model (eval mode is NOT set — caller decides).
    """
    from src.models.conv_vae import ConvVAE
    from src.models.conv_ae import ConvAE
    from src.models.conv_vqvae import ConvVQVAE

    data_cfg = config["data"]
    model_cfg = config["model"]
    model_type = config.get("model_type", "vae")

    if model_type == "vae":
        common = dict(
            T=data_cfg["T"],
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
            first_kernel=model_cfg.get("first_kernel", 25),
            kernel_size=model_cfg.get("kernel_size", 9),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
        )
        model = ConvVAE(
            **common,
            logvar_clip=tuple(model_cfg.get("logvar_clip", [-10, 10])),
        )
    elif model_type == "vqvae":
        vq_cfg = config.get("vq", {})
        model = ConvVQVAE(
            T=data_cfg["T"],
            channels=tuple(model_cfg["channels"]),
            first_kernel=model_cfg.get("first_kernel", 31),
            kernel_size=model_cfg.get("kernel_size", 11),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
            embedding_dim=int(vq_cfg.get("embedding_dim", model_cfg.get("embedding_dim", 64))),
            codebook_size=int(vq_cfg.get("codebook_size", model_cfg.get("codebook_size", 256))),
            commitment_cost=float(vq_cfg.get("commitment_cost", 0.25)),
        )
    else:
        common = dict(
            T=data_cfg["T"],
            latent_dim=model_cfg["latent_dim"],
            channels=tuple(model_cfg["channels"]),
            first_kernel=model_cfg.get("first_kernel", 25),
            kernel_size=model_cfg.get("kernel_size", 9),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
        )
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
    state = torch.load(path, map_location=map_loc, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model
