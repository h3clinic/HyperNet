"""Train our CNN from procedural pixels; no photographs or downloaded weights."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from hypernet.procedural_pretraining import MODES, train_procedural


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='detail_metric')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--triplets', type=int, default=16)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1 or args.triplets < 2 or args.threads < 1:
        parser.error('Need positive steps/threads and at least two triplets per step')
    torch.set_num_threads(args.threads)
    metadata = train_procedural(args.output, mode=args.mode, steps=args.steps,
                               triplets_per_step=args.triplets, seed=args.seed)
    print(json.dumps({key: metadata[key] for key in (
        'mode', 'steps', 'seed', 'rendered_image_exposures',
        'corrupted_view_exposures', 'natural_pretraining_images',
        'externally_provided_pretrained_weights', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
