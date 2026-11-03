"""Rare-glitch discovery search over the O3b candidate pool.

For each of five rare glitch types, the five v0.4 references (v1.1 train) are
enrolled and every pool image is scored by three label-free arms. The feature
vectors are built exactly as in v0.4's Bank (per-feature standardization of the
whole four-layer vector on the 600 unlabeled v1.1 train images, per-image z for
raw pixels); the only departures from v0.4 are (a) one duration, 0.5 s, and
(b) the mean of per-reference z-scores over the five references of a type.
Because a single 0.5 s view was never a v0.4 candidate, `--stage measure`
scores the same three single-view arms on the v0.5 O3b TEST set first, and the
freeze stage records those mean APs so the primary arm rests on a measured number.

  hybrid   z-fusion of raw pixels and the seed-1403 random CNN four-layer mix   <- primary
  random   seed-2026 random CNN four-layer mix
  trained  spectro_family epoch 8, coarse 6x6 grid

The search reads no label of any pool event. protocol.json is written before
the search and pins the pool, arrays, both scripts, the verifier's constants,
the truth tiers, the evaluated subset, the primary cell and the pass rule.

    python scripts/gravity_spy_discovery.py --stage measure ...   # single-view arms on v0.5 TEST
    python scripts/gravity_spy_discovery.py --stage freeze  ...   # protocol.json, no pool scores
    python scripts/gravity_spy_discovery.py --stage search  ...   # scores the pool, writes rankings
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
from gravity_spy_v02 import unit
import gravity_spy_v04 as v4
from prepare_gravity_spy_discovery import RARE, DURATION
import gravity_spy_discovery_verify as verify

ARMS = {'hybrid': '0.5 s single view: mean of per-reference z of raw pixels (per-image z) and random1403 four-layer mix (v0.4 per-feature rule)',
        'random': '0.5 s single view: random2026 four-layer mix', 'trained': '0.5 s single view: spectro_family_e8 coarse 6x6 grid'}
PRIMARY = {'arm': 'hybrid', 'k': 200, 'truth': 'new_majority', 'subset': 'pool events whose O3b ml_label != type',
           'types': ['Paired_Doves', '1400Ripples', 'Helix'], 'descriptive_only': ['Chirp', 'Wandering_Line'],
           'alpha': 0.05 / 3,
           'rule': ('POSITIVE only if, in at least 2 of the 3 primary types, the hypergeometric tail probability of the hybrid '
                    'top-200 new_majority count (within the evaluated subset, on that ranking\'s own covered events) is below '
                    '0.05/3 AND the hybrid new_majority@200 counted per 4 s group exceeds the CNN P(type) comparator\'s on the '
                    'same subset and K. Otherwise "not confirmed". Chirp is descriptive until an injection cross-match exists; '
                    'Wandering_Line has too few volunteer-majority events (4 pool-wide).')}
TOP_K = 200


def build_models(args, v04):
    family_path = args.trained / 'spectro_family' / 'epoch_08.pt'
    if sha256(family_path) != v04['models']['spectro_family_e8']:
        raise RuntimeError('trained checkpoint differs from v0.4')
    if sha256(args.runs / v4.RANDOM_CHECKPOINT[0]) != v4.RANDOM_CHECKPOINT[1]:
        raise RuntimeError('published random control differs')
    return {'random1403': v4.random_encoder(1403), 'random2026': v4.random_encoder(2026), 'family': PartDetector(family_path).model}


class Scorer:
    """v0.4-identical vectors; per-reference z against the 600 standardization images; mean over a type's references."""
    def __init__(self, models, ref_crops, ref_views, std_crops):
        self.models, self.ref_views = models, ref_views
        self.stats = {}
        std_f = self.raw(std_crops)
        for name in ('random1403', 'random2026', 'family'):
            self.stats[name] = (std_f[name].mean(0), std_f[name].std(0) + 1e-6)
        self.ref_v, std_v = self.vectors(self.raw(ref_crops)), self.vectors(std_f)
        self.background = {}
        for name in ('px', 'random1403', 'random2026', 'family'):
            for key, rows in ref_views.items():
                bg = (self.ref_v[name][rows] @ std_v[name].T).max(0); self.background[(name, key)] = (bg.mean(), bg.std() + 1e-6)

    def raw(self, crops):
        out = {'px': np.stack([pixel_vector(c) for c in crops])}
        for m in ('random1403', 'random2026'):
            out[m] = v4.model_readouts(layer_features(self.models[m], crops))['full_concat']
        out['family'] = layer_features(self.models['family'], crops)['coarse_grid6']
        return out

    def vectors(self, f):
        v = {'px': f['px']}
        for name in ('random1403', 'random2026', 'family'):
            v[name] = unit((f[name] - self.stats[name][0]) / self.stats[name][1])
        return v

    def arm_scores(self, crops, types, refs_per_type):
        v = self.vectors(self.raw(crops)); out = {a: {} for a in ARMS}
        for c in types:
            z = {}
            for name in ('px', 'random1403', 'random2026', 'family'):
                z[name] = np.mean([((self.ref_v[name][self.ref_views[(c, r)]] @ v[name].T).max(0) - self.background[(name, (c, r))][0]) / self.background[(name, (c, r))][1]
                                   for r in range(refs_per_type)], 0)
            out['hybrid'][c] = (z['px'] + z['random1403']) / 2; out['random'][c] = z['random2026']; out['trained'][c] = z['family']
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('measure', 'freeze', 'search'), required=True)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--v04', type=Path, default=ROOT / 'benchmarks/gravity_spy_v0.4'); parser.add_argument('--v05', type=Path, default=ROOT / 'benchmarks/gravity_spy_v0.5')
    parser.add_argument('--o3', type=Path, help='v0.5 cohort folder (for --stage measure)')
    parser.add_argument('--runs', type=Path, required=True); parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(); started = time.time()
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    v04 = json.loads((args.v04 / 'protocol.json').read_text())
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv'); by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    h5 = args.data / 'trainingsetv1d1.h5'
    all_rows = meta.index.to_numpy(); blank = {}
    for i, im in zip(all_rows, gs.load_split_images(h5, meta.loc[all_rows], '1.0')):
        blank[i] = bool(im.max() == im.min())
    all_ref_rows = {by_id[g] for c in v04['reference_gravityspy_ids'] for g in v04['reference_gravityspy_ids'][c]}
    pool_rows = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in all_ref_rows and not blank[i]])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool_rows, STANDARDIZE_IMAGES, replace=False))
    models = build_models(args, v04)

    def crops_h5(rows, d):
        return [crop_image(im)[0] for im in gs.load_split_images(h5, meta.loc[rows], d)]

    def enroll(types):
        refs = {c: [by_id[g] for g in v04['reference_gravityspy_ids'][c]] for c in types}
        ref_list, ref_views = [], {}
        for c, rr in refs.items():
            for r, crop in enumerate(crops_h5(rr, DURATION)):
                ref_views[(c, r)] = [len(ref_list), len(ref_list) + 1]; ref_list += [crop, np.ascontiguousarray(crop[:, ::-1])]
        return Scorer(models, ref_list, ref_views, crops_h5(std_rows, DURATION))

    if args.stage == 'measure':
        # Second read of the v0.5 O3b TEST arrays, for a descriptive single-view AP of each arm; disclosed in the protocol.
        if (args.output / 'single_view_ap.json').exists():
            raise FileExistsError('Already measured')
        v05 = json.loads((args.v05 / 'protocol.json').read_text()); classes = v05['evaluated_classes']
        scorer = enroll(classes)
        test = np.load(args.o3 / 'test_arrays.npz'); roster = pd.read_csv(args.o3 / 'test_roster.csv').set_index('gravityspy_id')
        labels = roster.loc[test['gravityspy_id'], 'target'].to_numpy()
        crops = [crop_image(gs.to_uint8_rgb(a))[0] for a in test[f'd{DURATION}']]
        per_arm = {a: {c: [] for c in classes} for a in ARMS}
        for start in range(0, len(crops), 2000):
            s = scorer.arm_scores(crops[start:start + 2000], classes, 5)
            for a in ARMS:
                for c in classes:
                    per_arm[a][c].append(s[a][c])
        result = {'format': 'hypernet.gravity_spy_discovery_single_view_ap.v0.2', 'cohort': 'v0.5 O3b TEST (second read, descriptive only)',
                  'duration': DURATION, 'classes': classes, 'v05_fused4_mean_ap': v05 and json.loads((args.v05 / 'results.json').read_text())['methods'] and
                  {a: json.loads((args.v05 / 'results.json').read_text())['methods'][b]['mean_ap'] for a, b in (('hybrid', 'best_hybrid'), ('random', 'best_random'), ('trained', 'best_trained'))},
                  'single_view_mean_ap': {}, 'per_class': {}}
        for a in ARMS:
            aps = {c: gs.average_precision(np.concatenate(per_arm[a][c]), labels == c) for c in classes}
            result['single_view_mean_ap'][a] = float(np.mean(list(aps.values()))); result['per_class'][a] = aps
        args.output.mkdir(parents=True, exist_ok=True); (args.output / 'single_view_ap.json').write_text(json.dumps(result, indent=1))
        print(json.dumps({k: result[k] for k in ('single_view_mean_ap', 'v05_fused4_mean_ap')}, indent=1)); return

    pool_record = json.loads((args.pool / 'pool.json').read_text()); arrays_manifest = json.loads((args.pool / 'arrays.json').read_text())
    inputs = {'pool.json': sha256(args.pool / 'pool.json'), 'pool_roster.csv': sha256(args.pool / 'pool_roster.csv'), 'arrays.json': sha256(args.pool / 'arrays.json'),
              'chunks': {c['file']: c['sha256'] for c in arrays_manifest['chunks']},
              'search_script_sha256': sha256(Path(__file__)), 'verify_script_sha256': sha256(ROOT / 'scripts/gravity_spy_discovery_verify.py'),
              'prepare_script_sha256': sha256(ROOT / 'scripts/prepare_gravity_spy_discovery.py'), 'v04_script_sha256': sha256(ROOT / 'scripts/gravity_spy_v04.py'),
              'single_view_ap.json': sha256(args.output / 'single_view_ap.json'), 'torch': torch.__version__}

    if args.stage == 'freeze':
        if (args.output / 'protocol.json').exists():
            raise FileExistsError('Protocol already frozen')
        picks_csv = args.output / 'pre_freeze_pool_volunteer_picks.csv'
        if not picks_csv.exists():
            raise FileNotFoundError('Move pool_volunteer_picks.csv into the output folder as pre_freeze_pool_volunteer_picks.csv first')
        single = json.loads((args.output / 'single_view_ap.json').read_text())
        pre = verify.pool_ceilings(picks_csv, args.pool / 'pool_roster.csv')
        protocol = {'format': 'hypernet.gravity_spy_discovery.v0.2', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'question': 'Does a one-reference search over the uncertain part of O3b concentrate volunteer-supported rare glitches that neither Gravity Spy CNN labeled as that type?',
                    'rare_types': list(RARE), 'references_per_type': 5, 'reference_gravityspy_ids': {c: v04['reference_gravityspy_ids'][c] for c in RARE},
                    'arms': ARMS, 'single_view_mean_ap_on_v05_test': single['single_view_mean_ap'], 'v05_fused4_mean_ap_for_contrast': single['v05_fused4_mean_ap'],
                    'primary': PRIMARY, 'top_k': TOP_K, 'duration': DURATION,
                    'score': 'v0.4 Bank vectors; per-reference z against 600 unlabeled v1.1 train images; mean over the five references of a type; descending stable sort, ties by roster order',
                    'pool': {k: pool_record[k] for k in ('rule', 'events', 'why_counts', 'time_groups_4s', 'roster_sha256')},
                    'arrays': {k: arrays_manifest.get(k) for k in ('events', 'dropped_count', 'blank_rule', 'blank_count')},
                    'verification': verify.declaration(),
                    'pre_freeze_disclosure': {'file': picks_csv.name, 'sha256': sha256(picks_csv), 'rows': int(sum(1 for _ in open(picks_csv)) - 1),
                                              'note': 'Per-event volunteer picks for the five rare types over the whole pool were extracted and seen before this protocol was written; '
                                                      'the type list, K and truth tiers were chosen with these ceilings known. The search itself reads no labels.',
                                              'pool_ceilings_under_frozen_rules': pre},
                    'inputs': inputs, 'standardization': f'{STANDARDIZE_IMAGES} unlabeled v1.1 train images, seed {STANDARDIZE_SEED}', 'labels_read_during_search': False}
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print(json.dumps({k: protocol[k] for k in ('single_view_mean_ap_on_v05_test', 'primary', 'arrays')}, indent=1)); print(json.dumps(pre, indent=1)); return

    protocol = json.loads((args.output / 'protocol.json').read_text())
    if (args.output / 'rankings.json').exists():
        raise FileExistsError('Search already run')
    if protocol['inputs'] != inputs:
        raise RuntimeError(f'Inputs changed since the freeze: {[k for k in inputs if inputs[k] != protocol["inputs"].get(k)]}')
    scorer = enroll(RARE)
    ids, scores = [], {a: {c: [] for c in RARE} for a in ARMS}
    for chunk in arrays_manifest['chunks']:
        a = np.load(args.pool / chunk['file']); ids.extend(a['gravityspy_id'].tolist())
        crops = [crop_image(np.repeat(im[:, :, None], 3, axis=2))[0] for im in a['image']]
        for start in range(0, len(crops), 2000):
            s = scorer.arm_scores(crops[start:start + 2000], RARE, 5)
            for arm in ARMS:
                for c in RARE:
                    scores[arm][c].append(s[arm][c])
        print(f'  {len(ids)} events scored at {time.time() - started:.0f}s', flush=True)
    ids = np.array(ids)
    rankings = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for c in RARE:
            s = np.concatenate(scores[arm][c]); order = np.argsort(-s, kind='stable')[:TOP_K]
            rankings[arm][c] = [{'gravityspy_id': str(ids[i]), 'rank': n + 1, 'score': float(s[i])} for n, i in enumerate(order)]
    np.savez_compressed(args.output / 'scores.npz', gravityspy_id=ids, **{f'{arm}|{c}': np.concatenate(scores[arm][c]).astype(np.float32) for arm in ARMS for c in RARE})
    (args.output / 'rankings.json').write_text(json.dumps({'format': protocol['format'], 'protocol_sha256': sha256(args.output / 'protocol.json'),
                                                           'events_scored': int(len(ids)), 'rankings': rankings, 'runtime_seconds': round(time.time() - started, 1)}, indent=1))
    print(json.dumps({'events_scored': int(len(ids))}))


if __name__ == '__main__':
    main()
