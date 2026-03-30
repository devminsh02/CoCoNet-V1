from __future__ import annotations

import copy
import csv
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
from tqdm import tqdm

from datasets import build_eval_dataset, build_train_dataset
from engine.evaluator import Evaluator
from losses import LossManager
from models import CODModel
from utils.common import AverageMeter, ensure_dir, freeze_batchnorm_modules, move_to_device, save_json, set_seed


def single_item_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return batch[0]


class Trainer:
    def __init__(self, cfg: Dict[str, Any], run_name: str, resume: Optional[str] = None) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.resume = resume
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(int(cfg['project'].get('seed', 3407)))
        self.train_cfg = cfg['train']
        self.data_cfg = cfg['data']
        self.results_cfg = cfg['paths']['results']
        self.amp_enabled = bool(self.train_cfg.get('amp', True)) and self.device.type == 'cuda'
        self.grad_clip_norm = float(self.train_cfg.get('grad_clip_norm', 0.0) or 0.0)
        self.accum_steps = int(self.train_cfg.get('accumulation_steps', 1))
        self.freeze_backbone_bn = bool(self.train_cfg.get('freeze_backbone_bn', False))
        self.freeze_all_bn = bool(self.train_cfg.get('freeze_all_bn', False))
        self.freeze_bn_affine = bool(self.train_cfg.get('freeze_bn_affine', False))
        self.backbone_lr_mult = float(self.train_cfg.get('backbone_lr_mult', 1.0))

        self.ckpt_dir = ensure_dir(Path(self.results_cfg['checkpoints']) / run_name)
        self.log_dir = ensure_dir(Path(self.results_cfg['logs']) / run_name)
        self.debug_dir = ensure_dir(Path(self.results_cfg['debug']) / run_name)
        self.metrics_dir = ensure_dir(Path(self.results_cfg['metrics']) / run_name)
        self.csv_log_path = self.log_dir / 'train_log.csv'
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.model = CODModel(cfg).to(self.device)
        self.loss_manager = LossManager(cfg).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_grad_scaler()
        self.train_loader = self._build_train_loader()
        self.val_loader = self._build_val_loader()
        self.evaluator = Evaluator(cfg=cfg, device=self.device, run_name=run_name)

        self.start_epoch = 1
        self.best_salpha = float('-inf')
        self.best_mae = float('inf')
        if resume:
            self._load_checkpoint(resume)

    def _build_grad_scaler(self):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            try:
                return torch.amp.GradScaler(self.device.type, enabled=self.amp_enabled)
            except TypeError:
                try:
                    return torch.amp.GradScaler(device_type=self.device.type, enabled=self.amp_enabled)
                except TypeError:
                    pass
        return torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    @contextmanager
    def _autocast(self):
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            try:
                with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                    yield
                return
            except TypeError:
                pass
        with torch.cuda.amp.autocast(enabled=self.amp_enabled):
            yield

    def _build_optimizer(self):
        opt_cfg = self.train_cfg['optimizer']
        name = str(opt_cfg.get('name', 'adamw')).lower()
        lr = float(opt_cfg.get('lr', 1e-4))
        wd = float(opt_cfg.get('weight_decay', 1e-4))
        backbone_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith('backbone.'):
                backbone_params.append(p)
            else:
                other_params.append(p)
        params = [
            {'params': backbone_params, 'lr': lr * self.backbone_lr_mult},
            {'params': other_params, 'lr': lr},
        ]
        if name == 'adamw':
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        if name == 'adam':
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        raise ValueError(f'Unsupported optimizer: {name}')

    def _build_scheduler(self):
        epochs = int(self.train_cfg.get('epochs', 80))
        warmup_epochs = int(self.train_cfg.get('warmup_epochs', 5))
        import math
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return max(1e-6, float(epoch + 1) / max(1, warmup_epochs))
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _loader_kwargs(self, train: bool) -> Dict[str, Any]:
        num_workers = int(self.data_cfg.get('num_workers', 0))
        return {'num_workers': num_workers, 'pin_memory': bool(self.data_cfg.get('pin_memory', True)), 'persistent_workers': num_workers > 0 and train}

    def _build_train_loader(self) -> DataLoader:
        dataset = build_train_dataset(self.cfg, split_alias=str(self.data_cfg.get('train_split', 'train_main')))
        return DataLoader(dataset, batch_size=int(self.train_cfg.get('batch_size', 1)), shuffle=True, drop_last=False, **self._loader_kwargs(train=True))

    def _build_val_loader(self) -> Optional[DataLoader]:
        split_alias = self.data_cfg.get('val_split')
        if split_alias is None or str(split_alias).lower() in {'', 'null', 'none'}:
            return None
        dataset = build_eval_dataset(self.cfg, split_alias=str(split_alias))
        return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=single_item_collate, **self._loader_kwargs(train=False))

    def _log_row(self, row: Dict[str, Any]) -> None:
        write_header = not self.csv_log_path.exists()
        with open(self.csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save_checkpoint(self, path: Path, epoch: int, extra: Optional[Dict[str, Any]] = None) -> None:
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if hasattr(self.scaler, 'state_dict') else {},
            'best_salpha': self.best_salpha,
            'best_mae': self.best_mae,
            'run_name': self.run_name,
            'cfg': self.cfg,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and hasattr(self.scaler, 'load_state_dict'):
            self.scaler.load_state_dict(ckpt['scaler'])
        self.start_epoch = int(ckpt.get('epoch', 0)) + 1
        self.best_salpha = float(ckpt.get('best_salpha', float('-inf')))
        self.best_mae = float(ckpt.get('best_mae', float('inf')))

    def _find_nonfinite_tensor(self, obj: Any, prefix: str = '') -> Optional[str]:
        if torch.is_tensor(obj):
            if not torch.isfinite(obj).all():
                return prefix or 'tensor'
            return None
        if isinstance(obj, dict):
            for k, v in obj.items():
                bad = self._find_nonfinite_tensor(v, f'{prefix}.{k}' if prefix else str(k))
                if bad is not None:
                    return bad
            return None
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                bad = self._find_nonfinite_tensor(v, f'{prefix}[{i}]' if prefix else f'[{i}]')
                if bad is not None:
                    return bad
            return None
        return None

    def _prepare_bn(self) -> None:
        if self.freeze_all_bn:
            freeze_batchnorm_modules(self.model, freeze_affine=self.freeze_bn_affine)
        elif self.freeze_backbone_bn:
            freeze_batchnorm_modules(self.model.backbone, freeze_affine=self.freeze_bn_affine)

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._prepare_bn()
        meters = {k: AverageMeter(k) for k in ['loss_total','loss_seg','loss_aux','loss_boundary','loss_affinity','loss_topology']}
        skipped_nonfinite = 0
        progress = tqdm(self.train_loader, desc=f'Train[{epoch}]', leave=False)
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            images = batch['image'].to(self.device, non_blocking=True)
            targets = move_to_device(batch['targets'], self.device)
            with self._autocast():
                outputs = self.model(images)
                bad_out = self._find_nonfinite_tensor(outputs)
                bad_tgt = self._find_nonfinite_tensor(targets)
                if bad_out is not None or bad_tgt is not None:
                    skipped_nonfinite += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                losses = self.loss_manager(outputs, targets)
                total_loss = losses['total'] / max(1, self.accum_steps)
            if not torch.isfinite(total_loss).all():
                skipped_nonfinite += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue
            if self.amp_enabled:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            if step % max(1, self.accum_steps) == 0:
                if self.grad_clip_norm > 0:
                    if self.amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                if self.amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            bs = images.size(0)
            meters['loss_total'].update(float(losses['total'].detach().item()), bs)
            if 'seg' in losses:
                meters['loss_seg'].update(float(losses['seg'].detach().item()), bs)
            if 'aux_seg' in losses:
                meters['loss_aux'].update(float(losses['aux_seg'].detach().item()), bs)
            if 'boundary' in losses:
                meters['loss_boundary'].update(float(losses['boundary'].detach().item()), bs)
            if 'affinity' in losses:
                meters['loss_affinity'].update(float(losses['affinity'].detach().item()), bs)
            if 'topology' in losses:
                meters['loss_topology'].update(float(losses['topology'].detach().item()), bs)
            progress.set_postfix({'loss': f'{meters["loss_total"].avg:.4f}', 'skip': skipped_nonfinite})
        stats = {k: v.avg for k, v in meters.items()}
        stats['skipped_nonfinite'] = float(skipped_nonfinite)
        return stats

    def fit(self) -> Dict[str, Any]:
        epochs = int(self.train_cfg.get('epochs', 80))
        for epoch in range(self.start_epoch, epochs + 1):
            train_stats = self._train_one_epoch(epoch)
            self.scheduler.step()
            lr = float(self.optimizer.param_groups[-1]['lr'])
            row = {'epoch': epoch, 'lr': lr, **train_stats}
            if self.val_loader is not None:
                val_stats = self.evaluator.evaluate(self.model, self.val_loader, dataset_tag='dev', save_vis=False, save_preds=False, vis_limit=0, desc=f'Val[{epoch}]')
                row.update(val_stats)
                salpha = float(val_stats.get('S_alpha', float('-inf')))
                mae = float(val_stats.get('MAE', float('inf')))
                if salpha > self.best_salpha:
                    self.best_salpha = salpha
                    self._save_checkpoint(self.ckpt_dir / 'best_salpha.pth', epoch, extra={'best_metric': 'S_alpha'})
                if mae < self.best_mae:
                    self.best_mae = mae
                    self._save_checkpoint(self.ckpt_dir / 'best_mae.pth', epoch, extra={'best_metric': 'MAE'})
            self._log_row(row)
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, epoch)
            self._save_checkpoint(self.ckpt_dir / 'last.pth', epoch)
            save_json(row, self.metrics_dir / f'epoch_{epoch:03d}.json')
        summary = {'run_name': self.run_name, 'device': str(self.device), 'epochs': epochs, 'best_salpha': None if self.best_salpha == float('-inf') else self.best_salpha, 'best_mae': None if self.best_mae == float('inf') else self.best_mae}
        save_json(summary, self.debug_dir / 'train_summary.json')
        self.writer.close()
        return summary
