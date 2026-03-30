from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-root', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    fieldnames: set[str] = {'run_name', 'dataset'}
    root = Path(args.metrics_root)
    if not root.exists():
        raise FileNotFoundError(f'Metrics root does not exist: {root}')

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for json_file in sorted(run_dir.glob('*.json')):
            if json_file.name.startswith('epoch_'):
                continue
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            row = {'run_name': run_dir.name, 'dataset': json_file.stem}
            row.update(data)
            row = _normalize_row(row)
            rows.append(row)
            fieldnames.update(row.keys())

    ordered_fields = ['run_name', 'dataset'] + sorted(k for k in fieldnames if k not in {'run_name', 'dataset'})
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in ordered_fields})
    print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
