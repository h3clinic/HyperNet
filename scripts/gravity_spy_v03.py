"""Gravity Spy v0.3: does training on top of locked random early layers add anything?

Four models were retrained (scripts/train_locked_early.py) on the v0.14 math-shape
or object-program curriculum, with stem+fine or stem+fine+middle locked at the
seed-1403 random start. v0.3 compares their trained-layer readouts with the
v0.2 winner (random early mix) under the v0.2 rules: DEV selection, frozen
protocol, one TEST pass. Same references, splits, metrics and 600-image
unlabeled scaling as v0.2. No spectrogram trains any weight.

    python scripts/gravity_spy_v03.py --stage select ...
    python scripts/gravity_spy_v03.py --stage test ...
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from hypernet import gravity_spy as gs
from hypernet.part_detector import PartDetector
from gravity_spy_benchmark import pixel_vector, sha256
from gravity_spy_diagnose import layer_features, STANDARDIZE_IMAGES, STANDARDIZE_SEED
from gravity_spy_v02 import unit, load_crops, reference_crops, score_all

RANDOM = ('random', 'random_v3_1403_statistics.pt', 'fc78ae5c4833fc533d12564d6ee1c9dd57e3ee7b27fa5eea21f1ccc515016231')
LOCKED = ('math_lock3', 'math_lock2', 'objects_lock3', 'objects_lock2')
GRID = ('stem_grid6', 'fine_grid6', 'middle_grid6', 'coarse_grid6')


def readout_names(model):
    """Readouts of one model; for locked models only those touching a trained block."""
    if model == 'random':
        return ['random_early_concat', 'random_full_concat']
    names = [f'{model}_coarse_grid6', f'{model}_coarse_pyramid', f'{model}_global', f'{model}_full_concat']
    if model.endswith('lock2'):
        names += [f'{model}_middle_grid6', f'{model}_early_concat']
    return names


def build(feats, model):
    out = {}
    for name in readout_names(model):
        part = name[len(model) + 1:]
        if part == 'early_concat':
            out[name] = unit(np.concatenate([feats[k] for k in GRID[:3]], 1))
        elif part == 'full_concat':
            out[name] = unit(np.concatenate([feats[k] for k in GRID], 1))
        else:
            out[name] = feats[part]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'test'), required=True)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--v02', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='hypernet_v14/checkpoints (random start)')
    parser.add_argument('--locked', type=Path, required=True, help='folder holding the four locked runs')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(); started = time.time()
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    v01 = json.loads((args.benchmark / 'protocol.json').read_text())
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv')
    by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    refs = {c: [by_id[g] for g in ids] for c, ids in v01['reference_gravityspy_ids'].items()}
    ref_rows = {i for rows in refs.values() for i in rows}
    pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in ref_rows])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))

    if sha256(args.runs / RANDOM[1]) != RANDOM[2]:
        raise ValueError('Random start differs from the v0.14 release')
    models, provenance = {'random': PartDetector(args.runs / RANDOM[1]).model}, {'random': RANDOM[2]}
    for name in LOCKED:
        record = json.loads((args.locked / name / 'locked.json').read_text())
        if not record['locked_blocks_unchanged_after_training']:
            raise ValueError(f'{name} locked blocks moved')
        path = args.locked / name / 'epoch_12.pt'
        models[name] = PartDetector(path).model; provenance[name] = {'epoch_12_sha256': sha256(path), 'locked': record['locked_blocks']}

    duration, k = v01['duration_seconds'], v01['references_per_class']
    classes, rare = v01['target_classes'], v01['rare_classes']
    std_crops = load_crops(args, meta, std_rows, duration)
    stats = {}
    for m, model in models.items():
        for key, value in layer_features(model, std_crops).items():
            stats[(m, key)] = (value.mean(0), value.std(0) + 1e-6)

    def readouts(crops):
        out = {}
        for m, model in models.items():
            feats = {key: unit((v - stats[(m, key)][0]) / stats[(m, key)][1]) for key, v in layer_features(model, crops).items()}
            out.update(build(feats, m))
        return out

    names = [n for m in models for n in readout_names(m)]
    trained_names = [n for m in LOCKED for n in readout_names(m)]
    ref_list, ref_views = reference_crops(args, meta, refs, duration)
    ref_feats = readouts(ref_list)
    dev_rows = meta.index[meta['sample_type'] == v01['dev_split']].to_numpy()
    dev_feats = readouts(load_crops(args, meta, dev_rows, duration))
    dev_labels = meta.loc[dev_rows, 'label'].to_numpy()
    dev_scores = score_all(ref_feats, ref_views, dev_feats, names)
    dev_ap = {n: gs.dev_mean_ap(dev_scores[n], dev_labels, np.ones(len(dev_rows), bool), classes, k) for n in names}

    v02 = json.loads((args.v02 / 'protocol.json').read_text())
    if abs(dev_ap['random_early_concat'] - v02['dev_mean_ap']['random_early_concat']) > 1e-12:
        raise RuntimeError('v0.2 winner does not reproduce on DEV')

    if args.stage == 'select':
        if args.output.exists():
            raise FileExistsError('Selection already frozen')
        best = max(trained_names, key=lambda n: dev_ap[n])
        protocol = {'format': 'hypernet.gravity_spy_oneshot.v0.3', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'test_scores_computed_before_freeze': False,
                    'question': 'Do trained layers on top of locked random early layers beat the v0.2 random early mix?',
                    'selection_rule': 'Highest DEV mean AP among readouts that use at least one trained block',
                    'selected': {'best_locked_readout': best},
                    'primary_contrast': 'best_locked_readout minus random_early_concat (v0.2 winner), TEST mean AP, class bootstrap',
                    'secondary_contrasts': 'each locked model full_concat minus random_full_concat (same blocks, coarse trained vs random)',
                    'dev_mean_ap': dev_ap, 'models': provenance, 'v01_protocol_sha256': sha256(args.benchmark / 'protocol.json'),
                    'v02_protocol_sha256': sha256(args.v02 / 'protocol.json'),
                    'test_reuse_note': 'TEST was scored once each for v0.1 and v0.2; this is its third use.',
                    'unlabeled_standardization_images': STANDARDIZE_IMAGES, 'weight_updates_on_spectrograms': 0}
        args.output.mkdir(parents=True)
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print(json.dumps({'selected': best, 'dev_mean_ap': {n: round(v, 4) for n, v in sorted(dev_ap.items(), key=lambda x: -x[1])}}, indent=2))
        return

    protocol_path = args.output / 'protocol.json'
    if (args.output / 'results.json').exists():
        raise FileExistsError('TEST already scored')
    protocol = json.loads(protocol_path.read_text())
    if any(protocol['dev_mean_ap'][n] != dev_ap[n] for n in names):
        raise RuntimeError('DEV scores changed since the freeze')
    test_rows = meta.index[meta['sample_type'] == v01['test_split']].to_numpy()
    test_scores = score_all(ref_feats, ref_views, readouts(load_crops(args, meta, test_rows, duration)), names)
    labels = np.concatenate([dev_labels, meta.loc[test_rows, 'label'].to_numpy()])
    dev = np.r_[np.ones(len(dev_rows), bool), np.zeros(len(test_rows), bool)]
    summary = {}
    for n in names:
        joined = {key: np.concatenate([dev_scores[n][key], test_scores[n][key]]) for key in dev_scores[n]}
        summary[n], _ = gs.evaluate_method(joined, labels, dev, ~dev, classes, rare, k, v01['false_alarm_rate_on_dev'])
    ap = lambda n: [summary[n]['per_class'][c]['ap'] for c in classes]
    old = json.loads((args.benchmark / 'results.json').read_text())['methods']
    best = protocol['selected']['best_locked_readout']
    contrasts = {'best_locked_minus_v02_winner': gs.paired_class_bootstrap(ap(best), ap('random_early_concat')),
                 'best_locked_minus_v01_raw_pixels': gs.paired_class_bootstrap(ap(best), [old['raw_pixels']['per_class'][c]['ap'] for c in classes])}
    for m in LOCKED:
        contrasts[f'{m}_full_concat_minus_random_full_concat'] = gs.paired_class_bootstrap(ap(f'{m}_full_concat'), ap('random_full_concat'))
    contrasts = {key: {'difference': d, 'class_bootstrap_95ci': ci} for key, (d, ci) in contrasts.items()}
    result = {'format': protocol['format'], 'protocol_sha256': sha256(protocol_path), 'selected': protocol['selected'],
              'methods': summary, 'contrasts': contrasts, 'runtime_seconds': round(time.time() - started, 1)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    short = lambda s: {key: round(v, 4) for key, v in s.items() if isinstance(v, float)}
    print(json.dumps({'selected': best, 'best': short(summary[best]), 'v02_winner': short(summary['random_early_concat']),
                      'mean_ap_all': {n: round(summary[n]['mean_ap'], 4) for n in names}, 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
