from __future__ import annotations

import torch
from torch import nn


class CausalLogMelGRU(nn.Module):
    """Causal GRU with a separate future-log-Mel regression head per horizon."""

    def __init__(self, n_mels: int, hidden_size: int, num_layers: int, horizons: list[int], dropout: float = 0.0):
        super().__init__()
        self.horizons = tuple(horizons)
        self.gru = nn.GRU(n_mels, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.heads = nn.ModuleDict({str(horizon): nn.Linear(hidden_size, n_mels) for horizon in self.horizons})

    def forward(self, frames: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        states, hidden = self.gru(frames, hidden)
        return {horizon: self.heads[str(horizon)](states) for horizon in self.horizons}, hidden
