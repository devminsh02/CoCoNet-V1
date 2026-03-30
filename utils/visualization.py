from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.common import ensure_dir


def load_rgb_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'))


def resize_map_to_size(map_np: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    map_np = np.asarray(map_np, dtype=np.float32)
    map_np = np.nan_to_num(map_np, nan=0.0, posinf=1.0, neginf=0.0)
    return cv2.resize(map_np, (w, h), interpolation=cv2.INTER_LINEAR)


def resize_rgb_to_size(image_rgb: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    return cv2.resize(np.asarray(image_rgb, dtype=np.uint8), (w, h), interpolation=cv2.INTER_AREA)


def fit_long_side_size(size_hw: tuple[int, int], max_long_side: int) -> tuple[int, int]:
    h, w = size_hw
    long_side = max(h, w)
    if max_long_side <= 0 or long_side <= max_long_side:
        return h, w
    scale = float(max_long_side) / float(long_side)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def save_uint8_gray(gray: np.ndarray, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    Image.fromarray(gray.astype(np.uint8)).save(path)


def save_rgb(image_rgb: np.ndarray, path: str | Path, quality: int = 95) -> None:
    ensure_dir(Path(path).parent)
    image = Image.fromarray(image_rgb.astype(np.uint8))
    params = {}
    suffix = Path(path).suffix.lower()
    if suffix in {'.jpg', '.jpeg'}:
        params = {'quality': quality}
    image.save(path, **params)


def to_uint8_prob(prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    return (prob * 255.0).round().astype(np.uint8)


def safe_norm_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    if x.ndim == 3:
        x = x.mean(axis=0)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-8)


def colorize_heatmap(prob: np.ndarray) -> np.ndarray:
    prob_u8 = to_uint8_prob(safe_norm_map(prob))
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def overlay_mask(image_rgb: np.ndarray, mask_prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    mask = np.asarray(mask_prob, dtype=np.float32)
    mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    color = np.array([255.0, 0.0, 0.0], dtype=np.float32)[None, None, :]
    out = image * (1.0 - alpha * mask) + color * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, prob: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32)
    heat = colorize_heatmap(prob).astype(np.float32)
    out = image * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def make_error_map(pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)
    canvas = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    canvas[pred & gt] = np.array([0, 255, 0], dtype=np.uint8)
    canvas[pred & (~gt)] = np.array([255, 0, 0], dtype=np.uint8)
    canvas[(~pred) & gt] = np.array([0, 0, 255], dtype=np.uint8)
    return canvas


def add_title(image_rgb: np.ndarray, title: str, bar_height: int = 24) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    canvas = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    canvas[bar_height:] = image_rgb
    cv2.putText(canvas, title, (6, int(bar_height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(items: Sequence[Tuple[str, np.ndarray]], num_cols: int = 4, cell_size: Optional[Tuple[int, int]] = None, pad: int = 6, bg_value: int = 20) -> np.ndarray:
    valid_items = [(t, i) for t, i in items if i is not None]
    if not valid_items:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if cell_size is None:
        cell_h, cell_w = valid_items[0][1].shape[:2]
    else:
        cell_h, cell_w = cell_size
    titled = []
    for title, img in valid_items:
        if img.shape[:2] != (cell_h, cell_w):
            img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        titled.append(add_title(img, title))
    tile_h, tile_w = titled[0].shape[:2]
    rows = int(np.ceil(len(titled) / num_cols))
    board = np.full((pad + rows * tile_h + max(0, rows - 1) * pad + pad, pad + num_cols * tile_w + max(0, num_cols - 1) * pad + pad, 3), bg_value, dtype=np.uint8)
    for idx, tile in enumerate(titled):
        r = idx // num_cols
        c = idx % num_cols
        y = pad + r * (tile_h + pad)
        x = pad + c * (tile_w + pad)
        board[y:y + tile_h, x:x + tile_w] = tile
    return board


def render_affinity_graph(image_rgb: np.ndarray, coords: np.ndarray, edge_index: np.ndarray, edge_valid_mask: np.ndarray, token_valid_mask: np.ndarray) -> np.ndarray:
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    coords = np.asarray(coords, dtype=np.float32)
    coords = np.nan_to_num(coords, nan=0.5, posinf=1.0, neginf=0.0)
    coords_xy = np.stack([coords[:, 1] * (w - 1), coords[:, 0] * (h - 1)], axis=-1).astype(np.int32)
    for is_valid, (src, dst) in zip(edge_valid_mask.tolist(), edge_index.T.tolist()):
        if not is_valid:
            continue
        cv2.line(canvas, tuple(coords_xy[src].tolist()), tuple(coords_xy[dst].tolist()), color=(0, 255, 255), thickness=1)
    for is_valid, pt in zip(token_valid_mask.tolist(), coords_xy.tolist()):
        color = (255, 0, 0) if is_valid else (128, 128, 128)
        cv2.circle(canvas, tuple(pt), radius=2, color=color, thickness=-1)
    return canvas


def save_branch_summary_board(sample_dir: str | Path, image_rgb: np.ndarray, gt_mask: np.ndarray, pred_prob: np.ndarray, branch_maps: Dict[str, Optional[np.ndarray]], affinity_graph_rgb: Optional[np.ndarray] = None, filename: str = '20_branch_summary_board.jpg') -> None:
    items: List[Tuple[str, np.ndarray]] = [
        ('Input', image_rgb),
        ('GT Overlay', overlay_mask(image_rgb, gt_mask)),
        ('Final Overlay', overlay_mask(image_rgb, pred_prob)),
        ('Error Map', make_error_map((pred_prob >= 0.5).astype(np.uint8), (gt_mask >= 0.5).astype(np.uint8))),
    ]
    title_map = {
        'coarse_logits': 'A Coarse',
        'fine_logits': 'A Fine',
        'objectness_map': 'A Objectness',
        'used_objectness_map': 'A Used Objectness',
        'uncertainty_map': 'A Uncertainty',
        'used_uncertainty_map': 'A Used Uncertainty',
        'boundary_prior': 'A Boundary Prior',
        'used_boundary_prior': 'A Used Boundary Prior',
        'a_feats_energy': 'A Feature Energy',
        'roi_mask': 'B ROI',
        'boundary_candidate_map': 'B Boundary Cand',
        'boundary_logits': 'B Boundary Refined',
        'closure_logits': 'B Closure',
        'b_feats_energy': 'B Feature Energy',
        'fusion_gate': 'Fusion Gate',
        'fused_feats_energy': 'Fused Feature Energy',
    }
    ordered = ['coarse_logits','fine_logits','objectness_map','used_objectness_map','uncertainty_map','used_uncertainty_map','boundary_prior','used_boundary_prior','a_feats_energy','roi_mask','boundary_candidate_map','boundary_logits','closure_logits','b_feats_energy','fusion_gate','fused_feats_energy']
    for key in ordered:
        val = branch_maps.get(key)
        if val is None:
            continue
        items.append((title_map[key], overlay_heatmap(image_rgb, val)))
    if affinity_graph_rgb is not None:
        items.append(('Affinity Graph', affinity_graph_rgb))
    board = make_contact_sheet(items, num_cols=4, cell_size=(image_rgb.shape[0], image_rgb.shape[1]))
    save_rgb(board, Path(sample_dir)/filename)


def save_debug_pack(sample_dir: str | Path, image_rgb: np.ndarray, gt_mask: np.ndarray, pred_prob: np.ndarray, branch_maps: Dict[str, Optional[np.ndarray]], save_summary_board: bool = True, affinity_graph_rgb: Optional[np.ndarray] = None, display_long_side: int = 1280) -> None:
    sample_dir = ensure_dir(sample_dir)
    disp_h, disp_w = fit_long_side_size((image_rgb.shape[0], image_rgb.shape[1]), display_long_side)
    image_rgb = resize_rgb_to_size(image_rgb, (disp_h, disp_w))
    gt_mask = resize_map_to_size(gt_mask, (disp_h, disp_w))
    pred_prob = resize_map_to_size(pred_prob, (disp_h, disp_w))
    resized_maps: Dict[str, Optional[np.ndarray]] = {}
    for key, value in branch_maps.items():
        resized_maps[key] = None if value is None else resize_map_to_size(value, (disp_h, disp_w))
    if affinity_graph_rgb is not None and affinity_graph_rgb.shape[:2] != (disp_h, disp_w):
        affinity_graph_rgb = resize_rgb_to_size(affinity_graph_rgb, (disp_h, disp_w))

    pred_bin = (np.nan_to_num(pred_prob, nan=0.0) >= 0.5).astype(np.uint8)
    gt_bin = (np.nan_to_num(gt_mask, nan=0.0) >= 0.5).astype(np.uint8)
    save_rgb(image_rgb, sample_dir/'00_input.jpg')
    save_uint8_gray(to_uint8_prob(gt_mask), sample_dir/'01_gt.png')
    save_rgb(overlay_mask(image_rgb, gt_mask), sample_dir/'02_gt_overlay.jpg')
    save_uint8_gray(to_uint8_prob(pred_prob), sample_dir/'03_final_pred.png')
    save_rgb(overlay_mask(image_rgb, pred_prob), sample_dir/'04_final_overlay.jpg')
    save_rgb(make_error_map(pred_bin, gt_bin), sample_dir/'05_error_fp_fn.png')

    ordered = [
        ('coarse_logits','06_a_coarse_heatmap.png','heat'),
        ('fine_logits','07_a_fine_heatmap.png','heat'),
        ('objectness_map','08_a_objectness_raw_heatmap.png','heat'),
        ('used_objectness_map','09_a_objectness_used_heatmap.png','heat'),
        ('uncertainty_map','10_a_uncertainty_raw_heatmap.png','heat'),
        ('used_uncertainty_map','11_a_uncertainty_used_heatmap.png','heat'),
        ('boundary_prior','12_a_boundary_prior_raw_heatmap.png','heat'),
        ('used_boundary_prior','13_a_boundary_prior_used_heatmap.png','heat'),
        ('a_feats_energy','14_a_feature_energy_heatmap.png','heat'),
        ('roi_mask','15_b_roi_mask.png','gray'),
        ('boundary_candidate_map','16_b_boundary_candidate_heatmap.png','heat'),
        ('boundary_logits','17_b_boundary_refined_heatmap.png','heat'),
        ('closure_logits','18_b_closure_heatmap.png','heat'),
        ('b_feats_energy','19_b_feature_energy_heatmap.png','heat'),
        ('fusion_gate','20_fusion_gate_heatmap.png','heat'),
        ('fused_feats_energy','21_fused_feature_energy_heatmap.png','heat'),
    ]
    for key, filename, kind in ordered:
        value = resized_maps.get(key)
        if value is None:
            continue
        value = safe_norm_map(value) if kind == 'heat' else np.clip(value, 0.0, 1.0)
        if kind == 'gray':
            save_uint8_gray(to_uint8_prob(value), sample_dir/filename)
        else:
            save_rgb(colorize_heatmap(value), sample_dir/filename)
    if affinity_graph_rgb is not None:
        save_rgb(affinity_graph_rgb, sample_dir/'22_affinity_graph.png')
    if save_summary_board:
        save_branch_summary_board(sample_dir, image_rgb, gt_mask, pred_prob, resized_maps, affinity_graph_rgb, filename='23_branch_summary_board.jpg')
