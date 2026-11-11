"""Anonymous VADiffAD release: SMAP main experiment only."""
import argparse
import json
from pathlib import Path
import numpy as np
from config.parser import load_config
from vadiffad.engine import run_single_smap


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate VADiffAD on prepared SMAP only')
    parser.add_argument('--config', default='config/smap.yaml')
    parser.add_argument('--data', default=None, help='Already-prepared SMAP.npz')
    parser.add_argument('--output', default=None)
    parser.add_argument('--seeds', default=None, help='Comma-separated independent SMAP training seeds')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = args.data or cfg.data_file
    root = Path(args.output or cfg.output_dir)
    seeds = tuple(map(int, args.seeds.split(','))) if args.seeds else cfg.seeds
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error('Use one or more distinct random seeds')
    reports = []
    for seed in seeds:
        report = run_single_smap(cfg, data, root/f'seed_{seed}', seed, args.device, args.threads)
        reports.append(report)
    summary = {'dataset': 'SMAP', 'seeds': list(seeds), 'runs': len(reports)}
    for metric in ('precision', 'recall', 'f1', 'average_precision', 'auroc'):
        values = [r['metrics'][metric] for r in reports if r['metrics'][metric] is not None]
        summary[metric] = {'mean': float(np.mean(values)) if values else None,
                           'sample_std': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    root.mkdir(parents=True, exist_ok=True)
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
