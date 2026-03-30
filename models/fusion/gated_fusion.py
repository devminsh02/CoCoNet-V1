from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from utils.common import ConvBNReLU, upsample_like


class GatedFusion(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.b_proj = ConvBNReLU(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Sequential(ConvBNReLU(channels * 2 + 3, channels), nn.Conv2d(channels, channels, 1), nn.Sigmoid())

    def forward(self, a_feat: torch.Tensor, b_feat: Optional[torch.Tensor], boundary_logits: Optional[torch.Tensor], closure_logits: Optional[torch.Tensor], uncertainty_map: Optional[torch.Tensor]) -> Dict[str, Optional[torch.Tensor]]:
        if b_feat is None:
            gate = torch.zeros_like(a_feat)
            return {'fused_feats': a_feat, 'fusion_gate': gate}
        boundary = upsample_like(boundary_logits, a_feat) if boundary_logits is not None else torch.zeros_like(a_feat[:, :1])
        closure = upsample_like(closure_logits, a_feat) if closure_logits is not None else torch.zeros_like(a_feat[:, :1])
        uncertainty = upsample_like(uncertainty_map, a_feat) if uncertainty_map is not None else torch.zeros_like(a_feat[:, :1])
        b_proj = self.b_proj(b_feat)
        gate = self.gate(torch.cat([a_feat, b_proj, boundary, closure, uncertainty], dim=1))
        return {'fused_feats': a_feat + gate * b_proj, 'fusion_gate': gate}
