from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class NullBranchB(nn.Module):
    def forward(self, *_args, **_kwargs) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        return {
            'pred': {'boundary_logits': None, 'closure_logits': None, 'affinity_logits': None},
            'aux': {'roi_mask': None, 'boundary_candidate_map': None, 'boundary_candidate_logits': None},
            'feat': {'b_feats': None, 'token_feats': None},
            'meta': {'fragment_coords': None, 'token_valid_mask': None, 'edge_index': None, 'edge_valid_mask': None},
        }
