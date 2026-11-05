"""Gravity Spy v0.2: read HyperNet out at earlier layers, selected on DEV, tested once.

The v0.1 diagnosis found the trained middle and deep blocks lose glitch
information. v0.2 changes no weights. It scores each candidate readout
(layer x pooling x trained-or-random weights) on DEV, freezes the winners, then
scores TEST once. References, splits and metrics are the same as v0.1.

Extra data budget: 600 unlabeled train-split spectrograms (not references) set
each feature's mean and scale. No labels from them are used.

    python scripts/gravity_spy_v02.py --stage select ...   # DEV only, writes protocol.json
    python scripts/gravity_spy_v02.py --stage test ...     # TEST once, writes results.json
"""
from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from hypernet import gravity_spy as gs
from hypernet.part_detector import PartDetector
from hypernet.part_encoder import crop_image
from gravity_spy_benchmark import pixel_vector, sha256
from gravity_spy_diagnose import layer_features, STANDARDIZE_IMAGES, STANDARDIZE_SEED

WEIGHTS = {'trained': ('parts_run03/epoch_12_statistics.pt', 'a64618f1d748acbb239c690abd2e5c2d26c1328162de5ac2b62a14394706013e'),
           'random': ('random_v3_1403_statistics.pt', 'fc78ae5c4833fc533d12564d6ee1c9dd57e3ee7b27fa5eea21f1ccc515016231')}
LAYERS = ('stem', 'fine', 'middle', 'coarse')
POOLS = ('grid6', 'pyramid')
EARLY_CONCAT = ('stem_grid6', 'fine_grid6', 'middle_grid6')
CANDIDATES = [f'{w}_{l}_{p}' for w in WEIGHTS for l in LAYERS for p in POOLS] + [f'{w}_early_concat' for w in WEIGHTS]
PRIMARY_RULES = {
    'best_cnn_readout': 'Highest DEV mean AP among all candidates (trained or random weights)',
    'best_trained_readout': 'Highest DEV mean AP among trained-weight candidates only',
}


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class Features:
    """Standardized readouts for one set of crops; the scaling comes only from the unlabeled train set."""
    def __init__(self, models, std_crops, pixel_std):
        self.models = models
        self.stats = {}
        for w, model in models.items():
            for key, value in layer_features(model, std_crops).items():
                self.stats[(w, key)] = (value.mean(0), value.std(0) + 1e-6)
        self.pixel_stats = (pixel_std.mean(0), pixel_std.std(0) + 1e-6)

    def readouts(self, crops):
        out = {}
        for w, model in self.models.items():
            feats = layer_features(model, crops)
            scaled = {k: unit((v - self.stats[(w, k)][0]) / self.stats[(w, k)][1]) for k, v in feats.items()}
            for l in LAYERS:
                for p in POOLS:
                    out[f'{w}_{l}_{p}'] = scaled[f'{l}_{p}']
            out[f'{w}_early_concat'] = unit(np.concatenate([scaled[k] for k in EARLY_CONCAT], 1))
        px = np.stack([pixel_vector(c) for c in crops])
        out['raw_pixels_standardized'] = unit((px - self.pixel_stats[0]) / self.pixel_stats[1])
        return out


def score_all(ref_feats, ref_views, item_feats, names):
    return {n: {key: (ref_feats[n][rows] @ item_feats[n].T).max(0) for key, rows in ref_views.items()} for n in names}


def setup(args):
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    v01 = json.loads((args.benchmark / 'protocol.json').read_text())
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv')
    by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    refs = {c: [by_id[g] for g in ids] for c, ids in v01['reference_gravityspy_ids'].items()}
    ref_rows = {i for rows in refs.values() for i in rows}
    pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in ref_rows])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))
    for w, (path, digest) in WEIGHTS.items():
        if sha256(args.runs / path) != digest:
            raise ValueError(f'{w} checkpoint differs from the published v0.14 release')
    models = {w: PartDetector(args.runs / path).model for w, (path, _) in WEIGHTS.items()}
    return v01, meta, refs, std_rows, models


def load_crops(args, meta, rows, duration):
    images = gs.load_split_images(args.data / 'trainingsetv1d1.h5', meta.loc[rows], duration)
    return [crop_image(im)[0] for im in images]


def reference_crops(args, meta, refs, duration):
    crops, views = [], {}
    for c, rows in refs.items():
        for r, crop in enumerate(load_crops(args, meta, rows, duration)):
            views[(c, r)] = [len(crops), len(crops) + 1]
            crops += [crop, np.ascontiguousarray(crop[:, ::-1])]
    return crops, views


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'test'), required=True)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(); started = time.time()
    v01, meta, refs, std_rows, models = setup(args)
    duration, k = v01['duration_seconds'], v01['references_per_class']
    classes, rare = v01['target_classes'], v01['rare_classes']
    std_crops = load_crops(args, meta, std_rows, duration)
    features = Features(models, std_crops, np.stack([pixel_vector(c) for c in std_crops]))
    ref_list, ref_views = reference_crops(args, meta, refs, duration)
    ref_feats = features.readouts(ref_list)
    names = CANDIDATES + ['raw_pixels_standardized']

    dev_rows = meta.index[meta['sample_type'] == v01['dev_split']].to_numpy()
    dev_feats = features.readouts(load_crops(args, meta, dev_rows, duration))
    dev_labels = meta.loc[dev_rows, 'label'].to_numpy(); dev_all = np.ones(len(dev_rows), bool)
    dev_scores = score_all(ref_feats, ref_views, dev_feats, names)
    dev_ap = {n: gs.dev_mean_ap(dev_scores[n], dev_labels, dev_all, classes, k) for n in names}

    if args.stage == 'select':
        if args.output.exists():
            raise FileExistsError('Selection already frozen; choose a new output folder')
        trained_only = [n for n in CANDIDATES if n.startswith('trained_')]
        winners = {'best_cnn_readout': max(CANDIDATES, key=lambda n: dev_ap[n]),
                   'best_trained_readout': max(trained_only, key=lambda n: dev_ap[n])}
        protocol = {'format': 'hypernet.gravity_spy_oneshot.v0.2', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'v01_protocol_sha256': sha256(args.benchmark / 'protocol.json'),
                    'test_scores_computed_before_freeze': False,
                    'candidates': CANDIDATES, 'baseline_also_scored': 'raw_pixels_standardized',
                    'selection_rules': PRIMARY_RULES, 'selected': winners,
                    'dev_mean_ap': dev_ap, 'checkpoint_sha256': {w: d for w, (_, d) in WEIGHTS.items()},
                    'unlabeled_standardization_images': {'count': STANDARDIZE_IMAGES, 'seed': STANDARDIZE_SEED, 'split': 'train, excluding references'},
                    'weight_updates': 0, 'spectrogram_images_used_in_weight_training': 0,
                    'reference_views': 'crop and its horizontal reflection; score is the max over views',
                    'metrics': 'Same as v0.1, from hypernet.gravity_spy.evaluate_method'}
        args.output.mkdir(parents=True)
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print(json.dumps({'selected': winners, 'dev_mean_ap': {n: round(v, 4) for n, v in sorted(dev_ap.items(), key=lambda x: -x[1])}}, indent=2))
        return

    protocol_path = args.output / 'protocol.json'
    if not protocol_path.exists():
        raise FileNotFoundError('Run --stage select first')
    if (args.output / 'results.json').exists():
        raise FileExistsError('TEST already scored; results are preserved')
    protocol = json.loads(protocol_path.read_text())
    drift = {n: (protocol['dev_mean_ap'][n], dev_ap[n]) for n in names if protocol['dev_mean_ap'][n] != dev_ap[n]}
    if drift:
        raise RuntimeError(f'DEV scores changed since the freeze: {drift}')

    test_rows = meta.index[meta['sample_type'] == v01['test_split']].to_numpy()
    test_feats = features.readouts(load_crops(args, meta, test_rows, duration))
    test_scores = score_all(ref_feats, ref_views, test_feats, names)
    # evaluate_method needs DEV and TEST side by side for its per-reference cutoffs.
    labels = np.concatenate([dev_labels, meta.loc[test_rows, 'label'].to_numpy()])
    dev = np.r_[np.ones(len(dev_rows), bool), np.zeros(len(test_rows), bool)]
    ids = np.concatenate([meta.loc[dev_rows, 'gravityspy_id'].to_numpy(), meta.loc[test_rows, 'gravityspy_id'].to_numpy()])
    summary = {}
    for n in names:
        joined = {key: np.concatenate([dev_scores[n][key], test_scores[n][key]]) for key in dev_scores[n]}
        summary[n], _ = gs.evaluate_method(joined, labels, dev, ~dev, classes, rare, k, v01['false_alarm_rate_on_dev'], item_ids=ids)
    old = json.loads((args.benchmark / 'results.json').read_text())['methods']
    baselines = {'v01_hypernet_v14': old['hypernet_v14'], 'v01_random_v14': old['random_v14'],
                 'v01_raw_pixels': old['raw_pixels'], 'raw_pixels_standardized': summary['raw_pixels_standardized']}
    contrasts = {}
    for rule, winner in protocol['selected'].items():
        for base, data in baselines.items():
            diff, ci = gs.paired_class_bootstrap([summary[winner]['per_class'][c]['ap'] for c in classes],
                                                 [data['per_class'][c]['ap'] for c in classes])
            contrasts[f'{rule}_minus_{base}_mean_ap'] = {'difference': diff, 'class_bootstrap_95ci': ci}
    result = {'format': protocol['format'], 'protocol_sha256': sha256(protocol_path), 'selected': protocol['selected'],
              'test_items': len(test_rows), 'methods': summary, 'v01_baselines_mean_ap': {b: d['mean_ap'] for b, d in baselines.items()},
              'contrasts': contrasts, 'note': 'Non-selected candidates are reported for completeness; only the two selected rules are primary.',
              'runtime_seconds': round(time.time() - started, 1)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    short = lambda s: {k2: round(v, 4) for k2, v in s.items() if isinstance(v, float)}
    print(json.dumps({'selected': protocol['selected'],
                      'primary': {r: short(summary[w]) for r, w in protocol['selected'].items()},
                      'raw_pixels_standardized': short(summary['raw_pixels_standardized']),
                      'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
