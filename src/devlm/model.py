from __future__ import annotations

import torch
from torch import nn


class CausalPhonemeGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_phonemes: int, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_phonemes)

    def forward(self, frames: torch.Tensor, hidden: torch.Tensor | None = None):
        states, hidden = self.gru(frames, hidden)
        return self.classifier(states), hidden
