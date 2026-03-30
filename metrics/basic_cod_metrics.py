from __future__ import annotations

from typing import Dict

import numpy as np


class BasicCODMetrics:
    def __init__(self) -> None:
        self.backend = 'mae_fallback'
        self._mae_values: list[float] = []
        try:
            import py_sod_metrics  # type: ignore
            self.backend = 'pysodmetrics'
            self.mae = py_sod_metrics.MAE()
            self.sm = py_sod_metrics.Smeasure()
            self.em = py_sod_metrics.Emeasure()
            self.fm = py_sod_metrics.Fmeasure()
            self.wfm = py_sod_metrics.WeightedFmeasure()
        except Exception:
            self.mae = self.sm = self.em = self.fm = self.wfm = None

    def step(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f'Shape mismatch between prediction {pred.shape} and GT {gt.shape}')
        if self.backend == 'pysodmetrics':
            self.mae.step(pred, gt)
            self.sm.step(pred, gt)
            self.em.step(pred, gt)
            self.fm.step(pred, gt)
            self.wfm.step(pred, gt)
        else:
            self._mae_values.append(float(np.abs(pred.astype(np.float32)/255.0 - gt.astype(np.float32)/255.0).mean()))

    def get_results(self) -> Dict[str, float | str]:
        if self.backend == 'pysodmetrics':
            sm = float(self.sm.get_results()['sm'])
            em_info = self.em.get_results()['em']
            fm_info = self.fm.get_results()['fm']
            wfm = float(self.wfm.get_results()['wfm'])
            mae = float(self.mae.get_results()['mae'])
            return {
                'S_alpha': sm,
                'E_phi': float(np.mean(em_info['curve'])),
                'E_phi_adp': float(em_info['adp']),
                'E_phi_max': float(np.max(em_info['curve'])),
                'F_beta_w': wfm,
                'F_beta_adp': float(fm_info['adp']),
                'F_beta_mean': float(np.mean(fm_info['curve'])),
                'F_beta_max': float(np.max(fm_info['curve'])),
                'MAE': mae,
                'metric_backend': self.backend,
            }
        mae = float(np.mean(self._mae_values)) if self._mae_values else 0.0
        return {'MAE': mae, 'metric_backend': self.backend}
