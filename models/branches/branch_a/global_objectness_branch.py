from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.branches.branch_a.boundary_prior_head import BoundaryPriorHead
from models.branches.branch_a.coarse_head import CoarseHead
from models.branches.branch_a.fine_head import FineHead
from utils.common import ConvBNReLU, sigmoid_entropy_from_logits, upsample_like


class GlobalObjectnessBranch(nn.Module):
    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.pyramid_fuse = nn.Sequential(ConvBNReLU(channels * 4, channels), ConvBNReLU(channels, channels))
        self.coarse_adapter = ConvBNReLU(channels, channels)
        self.objectness_head = nn.Conv2d(channels, 1, 1)
        self.coarse_head = CoarseHead(channels)
        self.fine_head = FineHead(channels)
        self.boundary_prior_head = BoundaryPriorHead(channels)

    def forward(self, neck_feats: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        p2 = neck_feats['p2']
        p3 = upsample_like(neck_feats['p3'], p2)
        p4 = upsample_like(neck_feats['p4'], p2)
        p5 = upsample_like(neck_feats['p5'], p2)
        fused = self.pyramid_fuse(torch.cat([p2, p3, p4, p5], dim=1))
        coarse_feat = self.coarse_adapter(neck_feats['p4'])
        coarse_logits = upsample_like(self.coarse_head(coarse_feat), p2)
        fine_logits = self.fine_head(fused)
        objectness_logits = self.objectness_head(fused)
        boundary_prior_logits = self.boundary_prior_head(fused)
        uncertainty_map = sigmoid_entropy_from_logits(fine_logits)
        return {
            'pred': {'coarse_logits': coarse_logits, 'fine_logits': fine_logits},
            'aux': {
                'objectness_map': torch.sigmoid(objectness_logits),
                'uncertainty_map': uncertainty_map,
                'boundary_prior': torch.sigmoid(boundary_prior_logits),
                'objectness_logits': objectness_logits,
                'boundary_prior_logits': boundary_prior_logits,
            },
            'feat': {'a_feats': fused, 'coarse_feat': coarse_feat},
        }
