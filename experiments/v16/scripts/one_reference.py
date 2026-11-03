"""Enroll exactly one image with the project's own mathematical-only CNN."""
from pathlib import Path
import argparse, json, os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.relational_matcher import FrozenRelations, RECIPES, score_vectors


def load(path):
    with Image.open(path) as image: return np.asarray(image.convert('RGB')).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--query', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', choices=RECIPES, default='mean')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous result files')
    if args.threshold is not None and not np.isfinite(args.threshold): parser.error('Finite threshold required')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    encoder = FrozenRelations(args.checkpoint, args.device)
    bank, references = encoder.enroll_many([load(args.reference)])
    query = encoder.encode(load(args.query))
    scores = {k: float(v[0]) for k, v in score_vectors(bank, query).items()}
    encoder.assert_unchanged()
    result = {'reference_images': 1, 'reference_rgb_sha256': references[0], 'query_rgb_sha256': query['rgb_sha256'],
              'scores': scores, 'selected_recipe': args.recipe, 'score': scores[args.recipe],
              'alarm': None if args.threshold is None else scores[args.recipe] >= args.threshold,
              'threshold': args.threshold, 'score_is_probability': False,
              'weight_updates': 0, 'provenance': encoder.provenance(),
              'scope': 'Whole-image structural similarity. No localization claim. A decision threshold needs independent domain calibration; absent one, no alarm decision is made.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'result': str(args.output), 'score': result['score'], 'alarm': result['alarm']}))


if __name__ == '__main__': main()
