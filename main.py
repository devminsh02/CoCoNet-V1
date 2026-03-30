from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_eval_dataset, generate_default_splits
from engine import Evaluator, Trainer
from models import CODModel
from utils.common import ensure_dir, save_json, timestamp_string
from utils.config import dump_yaml_config, load_yaml_config


def select_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def build_run_dirs(cfg: Dict, run_name: str) -> Dict[str, Path]:
    results_cfg = cfg['paths']['results']
    return {
        'checkpoints': ensure_dir(Path(results_cfg['checkpoints']) / run_name),
        'logs': ensure_dir(Path(results_cfg['logs']) / run_name),
        'metrics': ensure_dir(Path(results_cfg['metrics']) / run_name),
        'vis': ensure_dir(Path(results_cfg['vis']) / run_name),
        'predictions': ensure_dir(Path(results_cfg['predictions']) / run_name),
        'debug': ensure_dir(Path(results_cfg['debug']) / run_name),
    }


def command_prepare_splits(cfg: Dict, _args: argparse.Namespace) -> None:
    info = generate_default_splits(cfg, force=True)
    print(info)


def command_sanity_model(cfg: Dict, args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model = CODModel(cfg).to(device)
    model.eval()
    x = torch.randn(1, 3, int(cfg['train']['input_size']), int(cfg['train']['input_size']), device=device)
    with torch.no_grad():
        out = model(x)
    summary = {'pred': {}, 'aux': {}, 'meta': {}}
    for group in ['pred', 'aux', 'meta']:
        for k, v in out[group].items():
            summary[group][k] = list(v.shape) if torch.is_tensor(v) else None
    print(summary)


def command_train(cfg: Dict, args: argparse.Namespace) -> None:
    if not args.skip_prepare:
        generate_default_splits(cfg, force=False)
    trainer = Trainer(cfg=cfg, run_name=args.run_name, resume=args.resume)
    summary = trainer.fit()
    print(summary)


def command_eval(cfg: Dict, args: argparse.Namespace) -> None:
    if not args.skip_prepare:
        generate_default_splits(cfg, force=False)
    device = select_device('auto')
    model = CODModel(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    evaluator = Evaluator(cfg=cfg, device=device, run_name=args.run_name)
    datasets = args.datasets if args.datasets is not None else list(cfg['data']['test_splits'].keys())
    for alias in datasets:
        dataset = build_eval_dataset(cfg, split_alias=cfg['data']['test_splits'].get(alias, alias))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b[0], num_workers=int(cfg['data'].get('num_workers', 0)), pin_memory=bool(cfg['data'].get('pin_memory', True)))
        metrics = evaluator.evaluate(model, loader, dataset_tag=alias, save_vis=True, save_preds=True)
        print(f'[{alias}]', metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CCR-COD full project entrypoint')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=str, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p0 = sub.add_parser('prepare-splits', parents=[common])
    p1 = sub.add_parser('sanity-model', parents=[common])
    p1.add_argument('--device', type=str, default='auto')
    p2 = sub.add_parser('train', parents=[common])
    p2.add_argument('--run-name', type=str, required=True)
    p2.add_argument('--resume', type=str, default=None)
    p2.add_argument('--skip-prepare', action='store_true')
    p3 = sub.add_parser('eval', parents=[common])
    p3.add_argument('--run-name', type=str, required=True)
    p3.add_argument('--checkpoint', type=str, required=True)
    p3.add_argument('--datasets', nargs='*', default=None)
    p3.add_argument('--skip-prepare', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if args.command == 'prepare-splits':
        command_prepare_splits(cfg, args)
    elif args.command == 'sanity-model':
        command_sanity_model(cfg, args)
    elif args.command == 'train':
        command_train(cfg, args)
    elif args.command == 'eval':
        command_eval(cfg, args)
    else:
        raise ValueError(args.command)


if __name__ == '__main__':
    main()
