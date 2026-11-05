"""Retrain a v0.14 curriculum with the early CNN blocks locked at their random start.

The Gravity Spy diagnosis found the trained stem was harmless but training damaged
the fine/middle/coarse blocks, while the same blocks at random weights carried
the most glitch information. This wrapper runs an unmodified v0.14 trainer
(scripts/train_parts_v2.py or scripts/train_semantic_parts.py) with chosen
blocks set to requires_grad=False before the optimizer is built, so they keep
the seed-1403 random weights exactly. It then checks that they did not move.

    python scripts/train_locked_early.py --curriculum math_shapes --lock stem fine middle \
        --cache ../../outputs/hypernet_v14/data/synthetic_parts_v14_moderate.npz --output <new folder>
"""
from pathlib import Path
import argparse, hashlib, json, sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
CURRICULA = {'math_shapes': ('train_parts_v2', 'PartEncoderV2'),
             'object_programs': ('train_semantic_parts', 'PartEncoderV3')}
BLOCKS = ('stem', 'fine', 'middle', 'coarse')
# The unlocked run02/run03 started from these exact states.
EXPECTED_INITIAL = {'math_shapes': '5aef739be2a8f5bd31c8769f20021aade42864c2cb65f987223687b866cd1352',
                    'object_programs': '9e9c8c4f13449fffd66efcd03e05dad7a6a610d95ed64b5afaceb8da5ae60ae2'}
RUN02_DIAGNOSIS_SHA256 = '0413f8e05d6775b8e51d096929bae3c66e56827a91b8a488a11496d777cfdcb1'


def block_hash(state, blocks):
    h = hashlib.sha256()
    for key in sorted(k for k in state if k.split('.')[0] in blocks):
        h.update(key.encode()); h.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--curriculum', choices=CURRICULA, required=True)
    parser.add_argument('--lock', nargs='+', choices=BLOCKS, required=True)
    parser.add_argument('--cache', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    module_name, class_name = CURRICULA[args.curriculum]
    trainer = __import__(module_name)
    base = getattr(trainer, class_name); captured = {}

    class Locked(base):
        def __init__(self):
            super().__init__()
            captured['initial'] = {k: v.clone() for k, v in self.state_dict().items()}
            for name in args.lock:
                getattr(self, name).requires_grad_(False)

    setattr(trainer, class_name, Locked)
    redirect = None
    if args.curriculum == 'math_shapes':
        # train_parts_v2 only hashes this file into its admission record. It is not
        # in the public checkout; use the copy whose hash run02 originally recorded.
        missing = ROOT / 'experiments/run01_learning_diagnosis.json'
        original = ROOT.parents[1] / 'outputs/hypernet_v14/experiments/run01_learning_diagnosis.json'
        if not missing.exists():
            if hashlib.sha256(original.read_bytes()).hexdigest() != RUN02_DIAGNOSIS_SHA256:
                raise ValueError('run01 diagnosis copy differs from the one run02 recorded')
            redirect = str(original); plain_sha = trainer.sha
            trainer.sha = lambda p: plain_sha(original if Path(p) == missing else p)
    sys.argv = [module_name, '--cache', str(args.cache), '--output', str(args.output), '--threads', str(args.threads)]
    trainer.main()

    plan = json.loads((args.output / 'training_plan.json').read_text())
    if plan['initial_state_sha256'] != EXPECTED_INITIAL[args.curriculum]:
        raise RuntimeError('Random start differs from the original v0.14 run')
    final = torch.load(args.output / 'epoch_12.pt', weights_only=True)['state_dict']
    unchanged = block_hash(final, args.lock) == block_hash(captured['initial'], args.lock)
    trained = [b for b in BLOCKS if b not in args.lock]
    moved = {b: block_hash(final, [b]) != block_hash(captured['initial'], [b]) for b in trained}
    record = {'format': 'hypernet.locked_early_training.v0.3', 'curriculum': args.curriculum, 'trainer': f'scripts/{module_name}.py',
              'trainer_sha256': hashlib.sha256((ROOT / 'scripts' / f'{module_name}.py').read_bytes()).hexdigest(),
              'locked_blocks': args.lock, 'trained_blocks': trained + ['local', 'embedding', 'mask and task heads'],
              'initial_state_sha256': plan['initial_state_sha256'], 'locked_blocks_unchanged_after_training': unchanged,
              'trained_blocks_changed': moved, 'locked_block_sha256': block_hash(final, args.lock),
              'admission_diagnosis_file_redirected_to': redirect,
              'spectrograms_used': 0, 'natural_training_images': 0, 'borrowed_weights': False}
    (args.output / 'locked.json').write_text(json.dumps(record, indent=2))
    if not unchanged or not all(moved.values()):
        raise RuntimeError(f'Locking check failed: {record}')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
