"""Lightweight model EMA utilities."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy

import torch


class ModelEMA:
    """Exponential moving average over a model state dict."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow_state = copy.deepcopy(model.state_dict())
        for tensor in self.shadow_state.values():
            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                tensor.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        """Update the EMA weights from the current model."""
        model_state = model.state_dict()
        for name, value in model_state.items():
            shadow = self.shadow_state[name]
            if not isinstance(value, torch.Tensor):
                self.shadow_state[name] = value
                continue
            if torch.is_floating_point(value):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    def state_dict(self) -> dict:
        """Serialize EMA state."""
        return {
            "decay": self.decay,
            "shadow_state": self.shadow_state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore EMA state."""
        self.decay = float(state["decay"])
        self.shadow_state = state["shadow_state"]

    def averaged_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        """Return a clone of the EMA weights."""
        return OrderedDict(
            (name, tensor.detach().cpu().clone() if isinstance(tensor, torch.Tensor) else tensor)
            for name, tensor in self.shadow_state.items()
        )

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """Temporarily swap the model parameters with EMA weights."""
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow_state, strict=True)
        try:
            yield model
        finally:
            model.load_state_dict(backup, strict=True)
