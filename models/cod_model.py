from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from models.backbones import ResNet50Backbone
from models.branches.branch_a import GlobalObjectnessBranch
from models.branches.branch_b import ContourClosureBranch, NullBranchB
from models.decoders import RefinementDecoder
from models.fusion import GatedFusion, IdentityFusion
from models.necks import SimpleFPNNeck
from utils.common import upsample_to


class CODModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg['model']
        self.signal_switches = model_cfg.get('branch_a_signal_switches', {
            'objectness_map': True,
            'uncertainty_map': True,
            'boundary_prior': True,
        })
        self.backbone = ResNet50Backbone(pretrained=model_cfg['backbone'].get('pretrained', True), freeze_stages=model_cfg['backbone'].get('freeze_stages', 0))
        self.neck = SimpleFPNNeck({'c2': 256, 'c3': 512, 'c4': 1024, 'c5': 2048}, out_channels=model_cfg['neck'].get('out_channels', 256))
        self.branch_a = GlobalObjectnessBranch(channels=model_cfg['branch_a'].get('channels', 256))
        if model_cfg['branch_b'].get('enabled', True):
            self.branch_b = ContourClosureBranch(
                channels=model_cfg['branch_b'].get('channels', 256),
                roi_hidden_channels=model_cfg['branch_b'].get('roi', {}).get('hidden_channels', 64),
                grid_size=model_cfg['branch_b'].get('tokenizer', {}).get('grid_size', 16),
                token_valid_threshold=model_cfg['branch_b'].get('tokenizer', {}).get('valid_threshold', 0.05),
                k_neighbors=model_cfg['branch_b'].get('graph', {}).get('k_neighbors', 8),
                num_graph_layers=model_cfg['branch_b'].get('graph', {}).get('num_layers', 2),
                affinity_enabled=model_cfg['branch_b'].get('affinity', {}).get('enabled', False),
            )
            self.fusion = GatedFusion(channels=model_cfg['fusion'].get('channels', 256))
        else:
            self.branch_b = NullBranchB()
            self.fusion = IdentityFusion()
        self.decoder = RefinementDecoder(channels=model_cfg['decoder'].get('channels', 256))

    def _upsample_dict(self, xdict: Dict[str, Any], size: tuple[int, int]) -> Dict[str, Any]:
        out = {}
        for k, v in xdict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                out[k] = upsample_to(v, size)
            else:
                out[k] = v
        return out

    def _maybe_disable_signal(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.signal_switches.get(name, True):
            return tensor
        return torch.zeros_like(tensor)

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        input_size = x.shape[-2:]
        encoder_feats = self.backbone(x)
        neck_feats = self.neck(encoder_feats)
        a_out = self.branch_a(neck_feats)

        raw_objectness = a_out['aux']['objectness_map']
        raw_uncertainty = a_out['aux']['uncertainty_map']
        raw_boundary_prior = a_out['aux']['boundary_prior']
        used_objectness = self._maybe_disable_signal('objectness_map', raw_objectness)
        used_uncertainty = self._maybe_disable_signal('uncertainty_map', raw_uncertainty)
        used_boundary_prior = self._maybe_disable_signal('boundary_prior', raw_boundary_prior)

        b_out = self.branch_b(
            low_level_feat=neck_feats['p2'],
            high_level_feat=neck_feats['p4'],
            objectness_map=used_objectness,
            uncertainty_map=used_uncertainty,
            boundary_prior=used_boundary_prior,
        )
        fusion_out = self.fusion(
            a_feat=a_out['feat']['a_feats'],
            b_feat=b_out['feat']['b_feats'],
            boundary_logits=b_out['pred']['boundary_logits'],
            closure_logits=b_out['pred']['closure_logits'],
            uncertainty_map=used_uncertainty,
        )
        final_logits = self.decoder(fusion_out['fused_feats'])

        pred = {
            'final_logits': final_logits,
            'coarse_logits': a_out['pred']['coarse_logits'],
            'fine_logits': a_out['pred']['fine_logits'],
            'boundary_logits': b_out['pred']['boundary_logits'],
            'closure_logits': b_out['pred']['closure_logits'],
            'affinity_logits': b_out['pred']['affinity_logits'],
        }
        aux = {
            'objectness_map': raw_objectness,
            'used_objectness_map': used_objectness,
            'uncertainty_map': raw_uncertainty,
            'used_uncertainty_map': used_uncertainty,
            'boundary_prior': raw_boundary_prior,
            'used_boundary_prior': used_boundary_prior,
            'roi_mask': b_out['aux']['roi_mask'],
            'fusion_gate': fusion_out['fusion_gate'],
            'boundary_candidate_map': b_out['aux']['boundary_candidate_map'],
            'boundary_candidate_logits': b_out['aux']['boundary_candidate_logits'],
            'objectness_logits': a_out['aux']['objectness_logits'],
            'boundary_prior_logits': a_out['aux']['boundary_prior_logits'],
        }
        feat = {
            'encoder_feats': encoder_feats,
            'neck_feats': neck_feats,
            'a_feats': a_out['feat']['a_feats'],
            'coarse_feat': a_out['feat']['coarse_feat'],
            'b_feats': b_out['feat']['b_feats'],
            'token_feats': b_out['feat']['token_feats'],
            'fused_feats': fusion_out['fused_feats'],
        }
        meta = {
            'fragment_coords': b_out['meta']['fragment_coords'],
            'token_valid_mask': b_out['meta']['token_valid_mask'],
            'edge_index': b_out['meta']['edge_index'],
            'edge_valid_mask': b_out['meta']['edge_valid_mask'],
            'signal_switches': self.signal_switches,
        }
        return {
            'pred': self._upsample_dict(pred, input_size),
            'aux': self._upsample_dict(aux, input_size),
            'feat': feat,
            'meta': meta,
        }
