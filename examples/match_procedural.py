"""Enroll one real reference using an own procedurally pretrained CNN."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hypernet.data import load_rgb
from hypernet.models.procedural_reference import FrozenProceduralMatcher, ProceduralReferenceRouter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True, type=Path,
                        help='Checkpoint created by examples/pretrain_shapes.py')
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--query', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--adapt', action='store_true',
                        help='Adapt an independent copy with the one reference; default keeps weights frozen')
    parser.add_argument('--checkpoints', type=Path, default=Path('checkpoints/reference_adaptations'))
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.threads < 1 or args.steps < 1:
        parser.error('--threads and --steps must be positive')
    output = args.output.resolve()
    if output in {p.resolve() for p in (args.reference, args.query, args.pretrained)} or output.exists():
        parser.error('--output must be a new path distinct from input files')
    torch.set_num_threads(args.threads)
    model = (ProceduralReferenceRouter(args.pretrained, args.checkpoints, steps=args.steps,
             seed=args.seed, device=args.device) if args.adapt else
             FrozenProceduralMatcher(args.pretrained, device=args.device))
    result = model.predict(load_rgb(args.reference), load_rgb(args.query))
    report = {'prediction': result.to_dict(), 'provenance': model.provenance(),
              'interpretation': 'Uncalibrated pair score; not a probability or accuracy estimate',
              'threshold': None, 'query_weight_updates': 0,
              'training_scope': 'Own procedural pretraining plus one real reference; no borrowed weights'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'score': result.score, 'bbox_xyxy': result.bbox_xyxy,
                      'adapted': args.adapt, 'output': str(output)}))


if __name__ == '__main__':
    main()
