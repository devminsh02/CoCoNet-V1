from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.common import SampleRecord
from datasets.target_builder import TargetBuilder
from datasets.transforms import TrainPairTransform


class CODTrainDataset(Dataset):
    def __init__(self, records: List[SampleRecord], transform: TrainPairTransform, boundary_width: int = 3) -> None:
        self.records = records
        self.transform = transform
        self.target_builder = TargetBuilder(boundary_width=boundary_width)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        image = Image.open(rec.image_path).convert('RGB')
        mask_img = Image.open(rec.mask_path).convert('L')
        targets_np = self.target_builder.build(np.array(mask_img))
        boundary_img = Image.fromarray(targets_np['boundary']).convert('L')
        data = self.transform(image=image, mask=mask_img, boundary=boundary_img)
        return {
            'image': data['image'],
            'targets': {'mask': data['mask'], 'boundary': data['boundary']},
            'meta': {
                'dataset_name': rec.dataset_name,
                'sample_id': rec.sample_id,
                'image_path': rec.image_path,
                'mask_path': rec.mask_path,
            },
        }
