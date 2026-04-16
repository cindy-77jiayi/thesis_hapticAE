"""Shared data loading and model construction helpers for scripts."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import AudioSignalDataset
from .preprocessing import collect_audio_files, estimate_global_rms


def _read_file_list(path: str | None) -> list[str] | None:
    """Read newline-delimited audio paths, ignoring empty lines."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configured file list does not exist: {path}")
    with file_path.open("r", encoding="utf-8") as fp:
        entries = [line.strip() for line in fp if line.strip()]
    if not entries:
        raise ValueError(f"Configured file list is empty: {path}")
    return entries


def build_dataloaders(
    config: dict,
    data_dir: str,
    batch_size: int | None = None,
    full_dataset: bool = False,
) -> dict:
    """Build train/val DataLoaders (or a single full-dataset loader) from config.

    Args:
        config: Parsed YAML config dict.
        data_dir: Root directory of dataset audio files.
        batch_size: Override config batch_size if provided.
        full_dataset: If True, return a single loader over all files
            (used for latent extraction). Train/val loaders are omitted.

    Returns:
        Dict with keys: 'audio_files', 'global_rms', and either
        'train_loader'/'val_loader' or 'all_loader'.
    """
    data_cfg = config["data"]
    bs = batch_size or config.get("training", {}).get("batch_size", 32)
    seed = int(config.get("seed", 42))

    audio_files = collect_audio_files(
        data_dir,
        extensions=data_cfg.get("extensions"),
    )
    assert len(audio_files) > 0, f"No audio files found in {data_dir}"

    N = len(audio_files)
    perm = np.random.permutation(N)
    split = int(data_cfg["train_split"] * N)

    configured_train_files = _read_file_list(data_cfg.get("train_file_list"))
    configured_val_files = _read_file_list(data_cfg.get("val_file_list"))
    if (configured_train_files is None) != (configured_val_files is None):
        raise ValueError(
            "Both data.train_file_list and data.val_file_list must be set together",
        )

    if configured_train_files is not None and configured_val_files is not None:
        train_files = configured_train_files
        val_files = configured_val_files
    else:
        train_files = [audio_files[i] for i in perm[:split]]
        val_files = [audio_files[i] for i in perm[split:]]

    normalize_mode = data_cfg.get("normalize_mode", "global_rms")
    global_rms = (
        estimate_global_rms(train_files, n=200, sr_expect=data_cfg["sr"])
        if normalize_mode == "global_rms"
        else 1.0
    )

    ds_kwargs = dict(
        T=data_cfg["T"],
        sr_expect=data_cfg["sr"],
        global_rms=global_rms,
        scale=data_cfg["scale"],
        use_minmax=data_cfg.get("use_minmax", False),
        segment_mode=data_cfg.get("segment_mode", "energy"),
        min_segment_ratio=data_cfg.get("min_segment_ratio", 1.0),
        normalize_mode=normalize_mode,
        clip_range=tuple(data_cfg["clip_range"]) if data_cfg.get("clip_range") is not None else None,
    )

    result = {"audio_files": audio_files, "global_rms": global_rms}

    if full_dataset:
        full_files = audio_files
        if configured_train_files is not None and configured_val_files is not None:
            full_files = configured_train_files + configured_val_files
        result["audio_files"] = full_files
        all_ds = AudioSignalDataset(
            full_files,
            random_seek=False,
            sample_with_replacement=False,
            num_samples=len(full_files),
            seed=seed,
            **ds_kwargs,
        )
        result["all_loader"] = DataLoader(
            all_ds, batch_size=bs, shuffle=False, drop_last=False,
        )
    else:
        segment_mode = data_cfg.get("segment_mode", "energy")
        train_random_seek = data_cfg.get(
            "train_random_seek",
            segment_mode == "hapticgen",
        )
        train_sample_with_replacement = data_cfg.get(
            "train_sample_with_replacement",
            segment_mode == "hapticgen",
        )
        val_random_seek = data_cfg.get("val_random_seek", False)
        val_sample_with_replacement = data_cfg.get("val_sample_with_replacement", False)
        train_ds = AudioSignalDataset(
            train_files,
            random_seek=train_random_seek,
            sample_with_replacement=train_sample_with_replacement,
            num_samples=data_cfg.get("train_num_samples") or len(train_files),
            seed=seed,
            **ds_kwargs,
        )
        val_ds = AudioSignalDataset(
            val_files,
            random_seek=val_random_seek,
            sample_with_replacement=val_sample_with_replacement,
            num_samples=data_cfg.get("val_num_samples") or len(val_files),
            seed=seed + 10_000,
            **ds_kwargs,
        )
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
    from src.models.haptic_codec import HapticCodec

    data_cfg = config["data"]
    model_cfg = config["model"]
    model_type = config.get("model_type", "vae")

    common = dict(
        T=data_cfg["T"],
        channels=tuple(model_cfg["channels"]),
        first_kernel=model_cfg.get("first_kernel", 25),
        kernel_size=model_cfg.get("kernel_size", 9),
        activation=model_cfg.get("activation", "leaky_relu"),
        norm=model_cfg.get("norm", "group"),
    )

    if model_type == "vae":
        model = ConvVAE(
            **common,
            latent_dim=model_cfg["latent_dim"],
            logvar_clip=tuple(model_cfg.get("logvar_clip", [-10, 10])),
        )
    elif model_type == "ae":
        model = ConvAE(
            **common,
            latent_dim=model_cfg["latent_dim"],
        )
    elif model_type == "codec":
        codec_channels = tuple(model_cfg.get("channels", [32, 64, 128, 128, 64]))
        strides = tuple(model_cfg.get("strides", [5, 4, 4, 2, 2]))
        if len(codec_channels) != len(strides):
            codec_channels = (*codec_channels, int(model_cfg.get("code_dim", 64)))
        model = HapticCodec(
            T=data_cfg["T"],
            channels=codec_channels,
            strides=strides,
            first_kernel=model_cfg.get("first_kernel", 25),
            kernel_size=model_cfg.get("kernel_size", 9),
            residual_kernel_size=model_cfg.get("residual_kernel_size", 7),
            activation=model_cfg.get("activation", "leaky_relu"),
            norm=model_cfg.get("norm", "group"),
            code_dim=model_cfg.get("code_dim", 64),
            n_codebooks=model_cfg.get("n_codebooks", 4),
            codebook_size=model_cfg.get("codebook_size", 256),
            commitment_weight=model_cfg.get("commitment_weight", 0.25),
            control_dim=model_cfg.get("control_dim", 16),
            control_hidden=model_cfg.get("control_hidden", 128),
            metric_dim=model_cfg.get("metric_dim", 8),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

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
