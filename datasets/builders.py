from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets.cod_eval_dataset import CODEvalDataset
from datasets.cod_train_dataset import CODTrainDataset
from datasets.common import SampleRecord
from datasets.transforms import EvalImageTransform, TrainPairTransform
from utils.common import ensure_dir, save_json

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']
MASK_EXTS = ['.png', '.jpg', '.jpeg', '.bmp']

DATASET_TAG_TO_INFO = {
    'cod10k_train': {'cfg_key': ('cod10k', 'train_image_root', 'train_mask_root'), 'dataset_name': 'COD10K'},
    'cod10k_test': {'cfg_key': ('cod10k', 'test_image_root', 'test_mask_root'), 'dataset_name': 'COD10K'},
    'camo_train': {'cfg_key': ('camo', 'train_image_root', 'mask_root'), 'dataset_name': 'CAMO'},
    'camo_test': {'cfg_key': ('camo', 'test_image_root', 'mask_root'), 'dataset_name': 'CAMO'},
    'chameleon_test': {'cfg_key': ('chameleon', 'image_root', 'mask_root'), 'dataset_name': 'CHAMELEON'},
    'nc4k_test': {'cfg_key': ('nc4k', 'image_root', 'mask_root'), 'dataset_name': 'NC4K'},
}


def _list_stems(root: Path, exts: Sequence[str], require_non_empty: bool = True) -> List[str]:
    if not root.exists():
        raise FileNotFoundError(f'Dataset root does not exist: {root}')
    if not root.is_dir():
        raise NotADirectoryError(f'Expected a directory, got: {root}')
    exts = {e.lower() for e in exts}
    stems = sorted([p.stem for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts])
    if require_non_empty and not stems:
        raise RuntimeError(f'No image files with extensions {sorted(exts)} found under: {root}')
    return stems


def _resolve_file(root: Path, stem: str, exts: Sequence[str]) -> Path:
    for ext in exts:
        candidate = root / f'{stem}{ext}'
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Could not resolve file for stem={stem} in {root}')


def _save_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def parse_split_file(path: str | Path) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' not in line:
                raise ValueError(f'Invalid split line: {line}')
            tag, sample_id = line.split('\t', 1)
            entries.append((tag.strip(), sample_id.strip()))
    return entries


def build_records_from_split(cfg: Dict, split_path: str | Path) -> List[SampleRecord]:
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f'Split file does not exist: {split_path}')
    datasets_cfg = cfg['paths']['datasets']
    records: List[SampleRecord] = []
    entries = parse_split_file(split_path)
    if not entries:
        raise RuntimeError(f'Split file is empty: {split_path}')
    for dataset_tag, sample_id in entries:
        info = DATASET_TAG_TO_INFO[dataset_tag]
        ds_key, image_root_key, mask_root_key = info['cfg_key']
        image_root = Path(datasets_cfg[ds_key][image_root_key])
        mask_root = Path(datasets_cfg[ds_key][mask_root_key])
        image_path = _resolve_file(image_root, sample_id, IMAGE_EXTS)
        mask_path = _resolve_file(mask_root, sample_id, MASK_EXTS)
        records.append(SampleRecord(str(image_path), str(mask_path), info['dataset_name'], sample_id))
    if not records:
        raise RuntimeError(f'No records were built from split: {split_path}')
    return records


def generate_default_splits(cfg: Dict, force: bool = False) -> Dict[str, object]:
    datasets_cfg = cfg['paths']['datasets']
    data_cfg = cfg.get('data', {})
    dev_ratio = float(data_cfg.get('dev_ratio', 0.1))
    seed = int(cfg['project'].get('seed', 3407))
    splits_root = ensure_dir(datasets_cfg['splits_root'])

    mapping = {
        'cod10k_train': Path(datasets_cfg['cod10k']['train_image_root']),
        'cod10k_test': Path(datasets_cfg['cod10k']['test_image_root']),
        'camo_train': Path(datasets_cfg['camo']['train_image_root']),
        'camo_test': Path(datasets_cfg['camo']['test_image_root']),
        'chameleon_test': Path(datasets_cfg['chameleon']['image_root']),
        'nc4k_test': Path(datasets_cfg['nc4k']['image_root']),
    }
    counts: Dict[str, int] = {}
    files: Dict[str, str] = {}
    alias_to_filename = {
        'cod10k_train': 'cod10k_train_cam_3040.txt',
        'cod10k_test': 'cod10k_test_cam_2026.txt',
        'camo_train': 'camo_train_1000.txt',
        'camo_test': 'camo_test_250.txt',
        'chameleon_test': 'chameleon_test_76.txt',
        'nc4k_test': 'nc4k_test_4121.txt',
    }
    alias_lines: Dict[str, List[str]] = {}
    for alias, root in mapping.items():
        stems = _list_stems(root, IMAGE_EXTS, require_non_empty=True)
        counts[alias] = len(stems)
        lines = [f'{alias}\t{stem}' for stem in stems]
        alias_lines[alias] = lines
        out_path = splits_root / alias_to_filename[alias]
        files[alias] = out_path.name
        if force or not out_path.exists():
            _save_lines(out_path, lines)

    train_concat = alias_lines['cod10k_train'] + alias_lines['camo_train']
    if not train_concat:
        raise RuntimeError('Training split is empty after scanning COD10K/CAMO roots. Check dataset paths first.')
    rng = random.Random(seed)
    rng.shuffle(train_concat)
    dev_count = int(round(len(train_concat) * dev_ratio))
    if len(train_concat) > 1:
        dev_count = min(max(dev_count, 1), len(train_concat) - 1)
    else:
        dev_count = 0
    train_dev = train_concat[:dev_count]
    train_main = train_concat[dev_count:]
    extra = {
        'train_concat': ('cod_train_concat_4040.txt', train_concat),
        'train_main': ('cod_train_concat_main_3636.txt', train_main),
        'train_dev': ('cod_train_concat_dev_404.txt', train_dev),
    }
    for alias, (fname, lines) in extra.items():
        path = splits_root / fname
        files[alias] = path.name
        counts[alias] = len(lines)
        if force or not path.exists():
            _save_lines(path, lines)
    summary = {'splits_root': str(splits_root), 'files': files, 'counts': counts, 'dev_ratio': dev_ratio, 'seed': seed}
    save_json(summary, splits_root / 'split_summary.json')
    save_json(files, splits_root / 'split_index.json')
    return summary


def _alias_to_filename(split_root: Path) -> Dict[str, Path]:
    return {
        'cod10k_train': split_root / 'cod10k_train_cam_3040.txt',
        'cod10k_test': split_root / 'cod10k_test_cam_2026.txt',
        'camo_train': split_root / 'camo_train_1000.txt',
        'camo_test': split_root / 'camo_test_250.txt',
        'chameleon_test': split_root / 'chameleon_test_76.txt',
        'nc4k_test': split_root / 'nc4k_test_4121.txt',
        'train_concat': split_root / 'cod_train_concat_4040.txt',
        'train_main': split_root / 'cod_train_concat_main_3636.txt',
        'train_dev': split_root / 'cod_train_concat_dev_404.txt',
    }


def build_train_dataset(cfg: Dict, split_alias: str) -> CODTrainDataset:
    split_root = Path(cfg['paths']['datasets']['splits_root'])
    split_path = _alias_to_filename(split_root)[split_alias]
    records = build_records_from_split(cfg, split_path)
    transform = TrainPairTransform(
        input_size=int(cfg['train']['input_size']),
        hflip=bool(cfg['train']['augment'].get('hflip', True)),
        random_rescale=bool(cfg['train']['augment'].get('random_rescale', True)),
        random_crop=bool(cfg['train']['augment'].get('random_crop', True)),
        color_jitter=bool(cfg['train']['augment'].get('color_jitter', True)),
    )
    boundary_width = int(cfg.get('targets', {}).get('boundary_width', 3))
    return CODTrainDataset(records, transform=transform, boundary_width=boundary_width)


def build_eval_dataset(cfg: Dict, split_alias: str) -> CODEvalDataset:
    split_root = Path(cfg['paths']['datasets']['splits_root'])
    split_path = _alias_to_filename(split_root)[split_alias]
    records = build_records_from_split(cfg, split_path)
    transform = EvalImageTransform(input_size=int(cfg['eval'].get('input_size', cfg['train'].get('test_size', 352))))
    return CODEvalDataset(records, transform=transform)
