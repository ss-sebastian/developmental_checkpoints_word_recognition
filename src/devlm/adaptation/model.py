from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from devlm.model import CausalPhonemeGRU


CHECKPOINT_IDS = tuple(f"M{index:02d}" for index in range(1, 31))


class FrozenGRUEncoder(nn.Module):
    """Checkpoint-backed Phase 1 encoder that exposes detached final states."""

    def __init__(self, language_model: CausalPhonemeGRU, vocabulary: dict[str, int], checkpoint_config: dict):
        super().__init__()
        self.language_model = language_model
        self.vocabulary = dict(vocabulary)
        self.checkpoint_config = dict(checkpoint_config)
        self.hidden_dim = int(language_model.gru.hidden_size)
        self.language_model.eval()
        self.language_model.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        feature_width: int,
        device: torch.device | str = "cpu",
        expected_hidden_dim: int | None = None,
    ) -> "FrozenGRUEncoder":
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = payload["config"]
        vocabulary = {str(key): int(value) for key, value in payload["vocabulary"].items()}
        hidden_dim = int(config["hidden_size"])
        if expected_hidden_dim is not None and hidden_dim != expected_hidden_dim:
            raise ValueError(f"Checkpoint hidden_dim={hidden_dim}, expected {expected_hidden_dim}")
        model = CausalPhonemeGRU(
            feature_width, hidden_dim, int(config["num_layers"]), len(vocabulary),
            float(config.get("dropout", 0.0)),
        )
        model.load_state_dict(payload["model"])
        model.to(device)
        return cls(model, vocabulary, config)

    def train(self, mode: bool = True):
        super().train(mode)
        self.language_model.eval()
        return self

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 3 or lengths.ndim != 1 or len(frames) != len(lengths):
            raise ValueError("frames must be [batch, time, features] and lengths must be [batch]")
        if torch.any(lengths < 1) or torch.any(lengths > frames.shape[1]):
            raise ValueError("Every sequence length must fall within the padded frame tensor")
        states, _ = self.language_model.gru(frames.contiguous())
        row = torch.arange(len(frames), device=frames.device)
        final = states[row, lengths.to(frames.device, dtype=torch.long) - 1]
        return final.detach()

    def parameter_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.language_model.state_dict().items()}

    def assert_unchanged(self, snapshot: dict[str, torch.Tensor]) -> None:
        current = self.language_model.state_dict()
        if snapshot.keys() != current.keys() or any(
            not torch.equal(snapshot[name], value.detach().cpu()) for name, value in current.items()
        ):
            raise AssertionError("A frozen Phase 1 parameter changed during task adaptation")


def make_binary_readout(hidden_dim: int, seed: int) -> nn.Linear:
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Linear(hidden_dim, 1)


def task_loss() -> nn.BCEWithLogitsLoss:
    return nn.BCEWithLogitsLoss()


def instantiate_checkpoint_readouts(
    hidden_dim: int,
    seed: int,
    checkpoint_ids: Iterable[str] = CHECKPOINT_IDS,
) -> dict[str, nn.Linear]:
    identifiers = tuple(checkpoint_ids)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Checkpoint identifiers must be unique")
    return {identifier: make_binary_readout(hidden_dim, seed) for identifier in identifiers}


def readout_logits(encoder: FrozenGRUEncoder, readout: nn.Linear, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if readout.in_features != encoder.hidden_dim or readout.out_features != 1:
        raise ValueError("Readout must be exactly Linear(encoder.hidden_dim, 1)")
    return readout(encoder(frames, lengths)).squeeze(-1)
