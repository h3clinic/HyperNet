"""Train on one supplied reference and score one independent query, without pretraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hypernet.data import load_rgb
from hypernet.v08_registry import ALIASES, make_v08_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--query', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=ALIASES, default='v08_full')
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints'))
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        parser.error('--steps and --threads must be positive')
    output = args.output.resolve()
    if output in {args.reference.resolve(), args.query.resolve()}:
        parser.error('--output cannot overwrite an input image')
    if output.exists():
        parser.error('--output already exists; choose a new path')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    reference, query = load_rgb(args.reference), load_rgb(args.query)
    model = make_v08_models(args.checkpoints, steps=args.steps, seed=args.seed,
                           device=args.device)[args.model]
    result = model.predict(reference, query)
    report = {'model': args.model, 'prediction': result.to_dict(),
              'provenance': model.provenance(),
              'interpretation': 'Uncalibrated pair score; not a probability or accuracy estimate',
              'threshold': None, 'query_weight_updates': 0,
              'target_reference_images_per_model': 1, 'external_pretraining': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'model': args.model, 'score': result.score,
                      'bbox_xyxy': result.bbox_xyxy, 'output': str(output)}))


if __name__ == '__main__':
    main()
