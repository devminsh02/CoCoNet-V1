from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FragmentTokenizer(nn.Module):
    def __init__(self, grid_size: int = 16, valid_threshold: float = 0.05) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.valid_threshold = valid_threshold

    def forward(self, feat: torch.Tensor, boundary_prob: torch.Tensor, roi_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, _, _ = feat.shape
        g = self.grid_size
        pooled_feat = F.adaptive_avg_pool2d(feat, (g, g))
        pooled_boundary = F.adaptive_avg_pool2d(boundary_prob, (g, g))
        pooled_roi = F.adaptive_avg_pool2d(roi_mask, (g, g))
        token_score = pooled_boundary * pooled_roi
        tokens = pooled_feat.flatten(2).transpose(1, 2).contiguous()
        token_score = token_score.flatten(2).transpose(1, 2).contiguous()
        token_valid_mask = token_score.squeeze(-1) > self.valid_threshold
        ys = torch.linspace(0.0, 1.0, g, device=feat.device)
        xs = torch.linspace(0.0, 1.0, g, device=feat.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(1, g * g, 2).repeat(b, 1, 1)
        return {
            'tokens': tokens,
            'token_score': token_score,
            'token_valid_mask': token_valid_mask,
            'fragment_coords': coords,
        }
