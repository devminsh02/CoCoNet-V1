from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.basic_cod_metrics import BasicCODMetrics
from utils.common import ensure_dir, save_json
from utils.visualization import load_rgb_image, render_affinity_graph, resize_map_to_size, save_debug_pack, save_uint8_gray, to_uint8_prob


class Evaluator:
    def __init__(self, cfg: Dict[str, Any], device: torch.device, run_name: str) -> None:
        self.cfg = cfg
        self.device = device
        self.run_name = run_name
        self.pred_root = ensure_dir(Path(cfg['paths']['results']['predictions']) / run_name)
        self.vis_root = ensure_dir(Path(cfg['paths']['results']['vis']) / run_name)
        self.metrics_root = ensure_dir(Path(cfg['paths']['results']['metrics']) / run_name)
        self.eval_cfg = cfg['eval']
        self.feature_reduce = str(self.eval_cfg.get('feature_reduce', 'mean_abs'))
        self.vis_long_side = int(self.eval_cfg.get('vis_long_side', 1280))
        self.save_feature_maps = bool(self.eval_cfg.get('save_feature_maps', True))

    def _stable_sigmoid_np(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=20.0, neginf=-20.0)
        arr = np.clip(arr, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-arr))

    def _normalize_map(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        mn = float(arr.min())
        mx = float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        else:
            arr = np.zeros_like(arr)
        return arr.astype(np.float32)

    def _reduce_tensor_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        t = tensor.detach().float()
        if t.dim() == 4:
            if t.size(1) == 1:
                return t
            if self.feature_reduce == 'mean_abs':
                return t.abs().mean(dim=1, keepdim=True)
            if self.feature_reduce == 'l2':
                return torch.sqrt((t * t).sum(dim=1, keepdim=True) + 1e-8)
            if self.feature_reduce == 'max_abs':
                return t.abs().amax(dim=1, keepdim=True)
            return t.mean(dim=1, keepdim=True)
        if t.dim() == 3:
            if t.size(0) == 1:
                return t.unsqueeze(0)
            if self.feature_reduce == 'mean_abs':
                return t.abs().mean(dim=0, keepdim=True).unsqueeze(0)
            if self.feature_reduce == 'l2':
                return torch.sqrt((t * t).sum(dim=0, keepdim=True) + 1e-8).unsqueeze(0)
            if self.feature_reduce == 'max_abs':
                return t.abs().amax(dim=0, keepdim=True).unsqueeze(0)
            return t.mean(dim=0, keepdim=True).unsqueeze(0)
        if t.dim() == 2:
            return t.unsqueeze(0).unsqueeze(0)
        raise ValueError(f'Unsupported tensor dim: {t.dim()}')

    def _map_from_output(self, tensor: Optional[torch.Tensor], size_hw: tuple[int, int], apply_sigmoid: bool = False, normalize: bool = True) -> Optional[np.ndarray]:
        if tensor is None:
            return None
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[-1] if len(tensor) > 0 else None
            if tensor is None:
                return None
        if not torch.is_tensor(tensor):
            return None
        reduced = self._reduce_tensor_channels(tensor)
        reduced = F.interpolate(reduced, size=size_hw, mode='bilinear', align_corners=False)
        arr = reduced[0, 0].detach().cpu().numpy().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if apply_sigmoid:
            arr = self._stable_sigmoid_np(arr)
        elif normalize:
            arr = self._normalize_map(arr)
        return arr.astype(np.float32)

    def evaluate(self, model: torch.nn.Module, loader: DataLoader, dataset_tag: str, save_vis: bool = True, save_preds: bool = True, vis_limit: Optional[int] = None, desc: Optional[str] = None) -> Dict[str, float]:
        model.eval()
        recorder = BasicCODMetrics()
        vis_saved = 0
        if vis_limit is None:
            vis_limit = int(self.eval_cfg.get('save_vis_per_dataset', 30))
        iterator = tqdm(loader, desc=desc or f'Eval[{dataset_tag}]', leave=False)
        with torch.no_grad():
            for batch in iterator:
                image = batch['image'].unsqueeze(0).to(self.device, non_blocking=True)
                gt_mask = np.nan_to_num(np.asarray(batch['mask'], dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
                meta = batch['meta']
                sample_id = meta['sample_id']
                image_h, image_w = meta['orig_size']
                gt_h, gt_w = int(gt_mask.shape[0]), int(gt_mask.shape[1])

                outputs = model(image)
                final_logits = outputs['pred']['final_logits']
                pred_prob_metric = torch.sigmoid(F.interpolate(final_logits.float(), size=(gt_h, gt_w), mode='bilinear', align_corners=False))[0, 0].detach().cpu().numpy().astype(np.float32)
                pred_prob_metric = np.nan_to_num(pred_prob_metric, nan=0.0, posinf=1.0, neginf=0.0)
                pred_prob_metric = np.clip(pred_prob_metric, 0.0, 1.0)

                pred_u8 = to_uint8_prob(pred_prob_metric)
                gt_u8 = to_uint8_prob(gt_mask)
                recorder.step(pred=pred_u8, gt=gt_u8)
                if save_preds:
                    save_uint8_gray(pred_u8, ensure_dir(self.pred_root / dataset_tag) / f'{sample_id}.png')

                if save_vis and vis_saved < vis_limit:
                    image_rgb = load_rgb_image(meta['image_path'])
                    pred_prob_vis = torch.sigmoid(F.interpolate(final_logits.float(), size=(image_h, image_w), mode='bilinear', align_corners=False))[0, 0].detach().cpu().numpy().astype(np.float32)
                    pred_prob_vis = np.nan_to_num(pred_prob_vis, nan=0.0, posinf=1.0, neginf=0.0)
                    pred_prob_vis = np.clip(pred_prob_vis, 0.0, 1.0)
                    branch_maps = {
                        'coarse_logits': self._map_from_output(outputs['pred'].get('coarse_logits'), (image_h, image_w), apply_sigmoid=True, normalize=False),
                        'fine_logits': self._map_from_output(outputs['pred'].get('fine_logits'), (image_h, image_w), apply_sigmoid=True, normalize=False),
                        'objectness_map': self._map_from_output(outputs['aux'].get('objectness_map'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'used_objectness_map': self._map_from_output(outputs['aux'].get('used_objectness_map'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'uncertainty_map': self._map_from_output(outputs['aux'].get('uncertainty_map'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'used_uncertainty_map': self._map_from_output(outputs['aux'].get('used_uncertainty_map'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'boundary_prior': self._map_from_output(outputs['aux'].get('boundary_prior'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'used_boundary_prior': self._map_from_output(outputs['aux'].get('used_boundary_prior'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'roi_mask': self._map_from_output(outputs['aux'].get('roi_mask'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'boundary_candidate_map': self._map_from_output(outputs['aux'].get('boundary_candidate_map'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                        'boundary_logits': self._map_from_output(outputs['pred'].get('boundary_logits'), (image_h, image_w), apply_sigmoid=True, normalize=False),
                        'closure_logits': self._map_from_output(outputs['pred'].get('closure_logits'), (image_h, image_w), apply_sigmoid=True, normalize=False),
                        'fusion_gate': self._map_from_output(outputs['aux'].get('fusion_gate'), (image_h, image_w), apply_sigmoid=False, normalize=True),
                    }
                    if self.save_feature_maps:
                        branch_maps.update({
                            'a_feats_energy': self._map_from_output(outputs['feat'].get('a_feats'), (image_h, image_w), normalize=True),
                            'b_feats_energy': self._map_from_output(outputs['feat'].get('b_feats'), (image_h, image_w), normalize=True),
                            'fused_feats_energy': self._map_from_output(outputs['feat'].get('fused_feats'), (image_h, image_w), normalize=True),
                        })
                    else:
                        branch_maps.update({'a_feats_energy': None, 'b_feats_energy': None, 'fused_feats_energy': None})
                    affinity_graph_rgb = None
                    coords = outputs['meta'].get('fragment_coords')
                    edge_index = outputs['meta'].get('edge_index')
                    edge_valid_mask = outputs['meta'].get('edge_valid_mask')
                    token_valid_mask = outputs['meta'].get('token_valid_mask')
                    if coords is not None and edge_index is not None and edge_valid_mask is not None and token_valid_mask is not None:
                        affinity_graph_rgb = render_affinity_graph(image_rgb, coords[0].detach().cpu().numpy(), edge_index.detach().cpu().numpy(), edge_valid_mask[0].detach().cpu().numpy().astype(bool), token_valid_mask[0].detach().cpu().numpy().astype(bool))
                    save_debug_pack(self.vis_root / dataset_tag / sample_id, image_rgb, resize_map_to_size(gt_mask, (image_h, image_w)), pred_prob_vis, branch_maps, save_summary_board=bool(self.eval_cfg.get('save_branch_summary_board', True)), affinity_graph_rgb=affinity_graph_rgb, display_long_side=self.vis_long_side)
                    vis_saved += 1
        results = recorder.get_results()
        results['num_samples'] = len(loader.dataset)
        save_json(results, self.metrics_root / f'{dataset_tag}.json')
        return results

