"""Checkpoint and run-artifact helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime

import torch

from src.utils.config import dump_config


@dataclass(frozen=True)
class RunArtifacts:
    """Common artifact locations for a training run."""

    run_dir: str

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.run_dir, "best_model.pt")

    @property
    def last_checkpoint_path(self) -> str:
        return os.path.join(self.run_dir, "last_checkpoint.pt")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.run_dir, "metrics.npz")

    @property
    def history_path(self) -> str:
        return os.path.join(self.run_dir, "history.json")

    @property
    def config_path(self) -> str:
        return os.path.join(self.run_dir, "resolved_config.yaml")

    @property
    def split_manifest_path(self) -> str:
        return os.path.join(self.run_dir, "data_split.json")

    @property
    def preview_dir(self) -> str:
        return os.path.join(self.run_dir, "validation_samples")


def prepare_run_dir(config: dict) -> RunArtifacts:
    """Create the run directory and persist the resolved config."""
    output_dir = config.get("output_dir", "outputs")
    run_name = config.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    config["run_name"] = run_name
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    artifacts = RunArtifacts(run_dir=run_dir)
    dump_config(artifacts.config_path, config)
    return artifacts


class CheckpointManager:
    """Manage last/best/epoch checkpoints for a training run."""

    def __init__(self, artifacts: RunArtifacts, config: dict):
        self.artifacts = artifacts
        ckpt_cfg = config.get("checkpoint", {})
        self.save_last = bool(ckpt_cfg.get("save_last", True))
        self.save_best = bool(ckpt_cfg.get("save_best", True))
        self.save_every = int(ckpt_cfg.get("save_every", 0) or 0)
        self.keep_last = int(ckpt_cfg.get("keep_last", 0) or 0)

    def epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.artifacts.run_dir, f"checkpoint_epoch_{epoch:03d}.pt")

    def save_best_model(self, state_dict: dict) -> None:
        if self.save_best:
            torch.save(state_dict, self.artifacts.best_model_path)

    def save_training_checkpoint(self, state: dict, epoch: int) -> None:
        if self.save_last:
            torch.save(state, self.artifacts.last_checkpoint_path)
        if self.save_every > 0 and epoch % self.save_every == 0:
            torch.save(state, self.epoch_checkpoint_path(epoch))
            self._flush_stale_epoch_checkpoints()

    def load_training_checkpoint(self, path: str) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    def _flush_stale_epoch_checkpoints(self) -> None:
        if self.keep_last <= 0:
            return
        prefix = "checkpoint_epoch_"
        checkpoints = []
        for name in os.listdir(self.artifacts.run_dir):
            if not name.startswith(prefix) or not name.endswith(".pt"):
                continue
            stem = name[len(prefix) : -3]
            if stem.isdigit():
                checkpoints.append((int(stem), os.path.join(self.artifacts.run_dir, name)))
        checkpoints.sort(key=lambda item: item[0])
        for _, path in checkpoints[:-self.keep_last]:
            if os.path.exists(path):
                os.remove(path)


def save_history_json(path: str, history: list[dict]) -> None:
    """Persist per-epoch history as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
