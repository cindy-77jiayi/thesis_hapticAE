"""Builders for training-time components."""

from __future__ import annotations

import torch

from .ema import ModelEMA


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build the optimizer from config."""
    opt_cfg = config.get("optimizer", {})
    return torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg.get("lr", 2e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-5),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build the learning-rate scheduler from config."""
    sched_cfg = config.get("scheduler", {})
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 15),
    )


def build_ema(model: torch.nn.Module, config: dict) -> ModelEMA | None:
    """Optionally build an EMA wrapper around the model."""
    ema_cfg = config.get("ema", {})
    if not ema_cfg.get("use", False):
        return None
    return ModelEMA(model, decay=float(ema_cfg.get("decay", 0.999)))

