from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class IdentityFusion(nn.Module):
    def forward(self, a_feat: torch.Tensor, b_feat: Optional[torch.Tensor] = None, boundary_logits: Optional[torch.Tensor] = None, closure_logits: Optional[torch.Tensor] = None, uncertainty_map: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        return {'fused_feats': a_feat, 'fusion_gate': torch.zeros_like(a_feat)}
