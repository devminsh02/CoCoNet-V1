from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.common import SampleRecord
from datasets.transforms import EvalImageTransform


class CODEvalDataset(Dataset):
    def __init__(self, records: List[SampleRecord], transform: EvalImageTransform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert('RGB')
        mask_img = Image.open(rec.mask_path).convert('L')
        width, height = image.size
        gt_mask = np.asarray(mask_img, dtype=np.float32) / 255.0
        image_tensor = self.transform(image)
        return {
            'image': image_tensor,
            'mask': gt_mask,
            'meta': {
                'dataset_name': rec.dataset_name,
                'sample_id': rec.sample_id,
                'image_path': rec.image_path,
                'mask_path': rec.mask_path,
                'orig_size': (height, width),
            },
        }
