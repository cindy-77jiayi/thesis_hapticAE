"""Shared building blocks for Conv1D encoder-decoder architectures."""

import torch.nn as nn


def group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


def make_activation(name: str = "leaky_relu") -> nn.Module:
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)
    return nn.ReLU()


def clone_activation(module: nn.Module) -> nn.Module:
    """Create a fresh activation layer with the same configuration."""
    if isinstance(module, nn.LeakyReLU):
        return nn.LeakyReLU(module.negative_slope)
    if isinstance(module, nn.ReLU):
        return nn.ReLU()
    return type(module)()


def make_norm(name: str = "group"):
    """Return a norm constructor: callable(channels) -> nn.Module."""
    if name == "group":
        return group_norm
    return nn.BatchNorm1d
