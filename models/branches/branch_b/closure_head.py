from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class ClosureHead(nn.Module):
    def __init__(self, channels: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True), nn.Linear(channels, 1))

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mlp(tokens)
        b, n, _ = logits.shape
        g = self.grid_size
        map_logits = logits.transpose(1, 2).reshape(b, 1, g, g)
        return {'closure_token_logits': logits, 'closure_map_logits': map_logits}
