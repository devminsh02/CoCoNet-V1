from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.branches.branch_b.affinity_head import AffinityHead
from models.branches.branch_b.boundary_candidate_head import BoundaryCandidateHead
from models.branches.branch_b.closure_head import ClosureHead
from models.branches.branch_b.fragment_tokenizer import FragmentTokenizer
from models.branches.branch_b.graph_reasoner import GraphReasoner
from models.branches.branch_b.roi_gating import SoftROIGating
from utils.common import ConvBNReLU, upsample_like


class ContourClosureBranch(nn.Module):
    def __init__(self, channels: int = 256, roi_hidden_channels: int = 64, grid_size: int = 16, token_valid_threshold: float = 0.05, k_neighbors: int = 8, num_graph_layers: int = 2, affinity_enabled: bool = True) -> None:
        super().__init__()
        self.affinity_enabled = affinity_enabled
        self.roi_gating = SoftROIGating(hidden_channels=roi_hidden_channels)
        self.feature_fuse = nn.Sequential(ConvBNReLU(channels * 2, channels), ConvBNReLU(channels, channels))
        self.boundary_candidate_head = BoundaryCandidateHead(channels)
        self.tokenizer = FragmentTokenizer(grid_size=grid_size, valid_threshold=token_valid_threshold)
        self.graph_reasoner = GraphReasoner(channels=channels, k_neighbors=k_neighbors, num_layers=num_graph_layers)
        self.closure_head = ClosureHead(channels=channels, grid_size=grid_size)
        self.boundary_refine = nn.Sequential(ConvBNReLU(channels + 1, channels), nn.Conv2d(channels, 1, 1))
        self.affinity_head = AffinityHead(channels=channels)

    def forward(self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor, objectness_map: torch.Tensor, uncertainty_map: torch.Tensor, boundary_prior: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        high_up = upsample_like(high_level_feat, low_level_feat)
        base_feat = self.feature_fuse(torch.cat([low_level_feat, high_up], dim=1))
        roi_mask = self.roi_gating(objectness_map, uncertainty_map, boundary_prior)
        gated_feat = base_feat * (1.0 + roi_mask)
        boundary_candidate_logits = self.boundary_candidate_head(gated_feat)
        boundary_candidate_prob = torch.sigmoid(boundary_candidate_logits)
        tok = self.tokenizer(gated_feat, boundary_candidate_prob, roi_mask)
        graph = self.graph_reasoner(tok['tokens'], tok['fragment_coords'], tok['token_valid_mask'])
        closure = self.closure_head(graph['tokens'])
        closure_up = upsample_like(closure['closure_map_logits'], gated_feat)
        boundary_logits = self.boundary_refine(torch.cat([gated_feat, closure_up], dim=1))
        affinity_logits = self.affinity_head(graph['tokens'], tok['fragment_coords'], graph['edge_index']) if self.affinity_enabled else None
        return {
            'pred': {'boundary_logits': boundary_logits, 'closure_logits': closure_up, 'affinity_logits': affinity_logits},
            'aux': {'roi_mask': roi_mask, 'boundary_candidate_map': boundary_candidate_prob, 'boundary_candidate_logits': boundary_candidate_logits},
            'feat': {'b_feats': gated_feat, 'token_feats': graph['tokens']},
            'meta': {
                'fragment_coords': tok['fragment_coords'],
                'token_valid_mask': tok['token_valid_mask'],
                'edge_index': graph['edge_index'],
                'edge_valid_mask': graph['edge_valid_mask'],
            },
        }
