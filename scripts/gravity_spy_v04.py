"""Gravity Spy v0.4: does a spectrogram-like curriculum with within-family resampling make
learned features beat random ones?

Models: spectro_family (new data, family objective), spectro_instance (new data,
old run02 objective), and FIVE random starts of the same architecture (the
v0.1-v0.3 control used one seed). Candidate epochs 4, 8, 12.
Readouts per model: global embedding, coarse 6x6 grid, four-layer grid mix;
each scored on the 1.0 s view alone and as a z-fused mean over the four
durations (z-statistics from each reference's scores against 600 unlabeled
train spectrograms). Controls: raw pixels (per-image z, as in v0.1), an
untrained wavelet scattering transform in two normalizations, and a hybrid of
raw pixels with the seed-1403 random four-layer mix.

Protocol changes from v0.1-v0.3, all decided on DEV before TEST is read:
  * exactly constant ("blank") spectrograms are excluded from the haystack, and
    a blank reference is replaced by a seeded draw from the same class;
  * normalization is per arm: per-feature standardization for CNN features,
    per-image z for raw pixels, both variants for scattering.

Selection on DEV only; protocol.json is written before TEST is read; TEST is
scored once. Same references (except replaced blanks), splits and metrics as
v0.1-v0.3.

    python scripts/gravity_spy_v04.py --stage select ...
    python scripts/gravity_spy_v04.py --stage test ...
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
from hypernet.part_encoder import crop_image
from hypernet.part_encoder_v2 import PartEncoderV2
from gravity_spy_benchmark import pixel_vector, sha256
from gravity_spy_diagnose import layer_features, STANDARDIZE_IMAGES, STANDARDIZE_SEED
from gravity_spy_v02 import unit

RANDOM_CHECKPOINT = ('random_v3_1403_statistics.pt', 'fc78ae5c4833fc533d12564d6ee1c9dd57e3ee7b27fa5eea21f1ccc515016231')
RANDOM_SEEDS = (1403, 2026, 7, 11, 23)
TRAINED = ('spectro_family', 'spectro_instance')
EPOCHS = (4, 8, 12)
GRID = ('stem_grid6', 'fine_grid6', 'middle_grid6', 'coarse_grid6')
READOUTS = ('global', 'coarse_grid6', 'full_concat')
SCORINGS = ('1.0s', 'fused4')
FAR = 0.01
BLANK_RULE = 'image max minus min below 1e-6 on the 1.0 s view'


def scattering():
    # The package entry point also imports a 3D module that fails on current SciPy; use the 2D front end directly.
    from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
    return ScatteringTorch2D(J=3, shape=(96, 96), L=8)


def random_encoder(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed); model = PartEncoderV2()
    return model.eval().requires_grad_(False)


def model_readouts(feats):
    return {'global': feats['global'], 'coarse_grid6': feats['coarse_grid6'],
            'full_concat': unit(np.concatenate([feats[g] for g in GRID], 1))}


@torch.inference_mode()
def scat_features(scat, crops, batch=64):
    out = []
    for start in range(0, len(crops), batch):
        gray = torch.from_numpy(np.stack(crops[start:start + batch])[..., 0]).float() / 255
        out.append(scat(gray).flatten(1).double().numpy())
    return np.concatenate(out)


def per_image_z(x):
    x = x - x.mean(1, keepdims=True); x = x / (x.std(1, keepdims=True) + 1e-6)
    return unit(x)


class Bank:
    """Feature vectors for one duration: arm -> (normalization, (ref rows, std rows, item rows))."""
    def __init__(self, models, scat, ref_crops, std_crops, item_crops):
        self.arms = {}
        for mname, model in models.items():
            parts = [layer_features(model, c) for c in (ref_crops, std_crops, item_crops)]
            for r in READOUTS:
                self.arms[f'{mname}|{r}'] = ('per_feature', tuple(model_readouts(p)[r] for p in parts))
        px = lambda cs: np.stack([pixel_vector(c) for c in cs])  # already per-image z and unit norm
        self.arms['raw_pixels'] = ('none', tuple(px(c) for c in (ref_crops, std_crops, item_crops)))
        sc = tuple(scat_features(scat, c) for c in (ref_crops, std_crops, item_crops))
        self.arms['scattering_perfeature'] = ('per_feature', sc)
        self.arms['scattering_perimage'] = ('per_image_log', sc)

    def scores(self, ref_views):
        out = {}
        for name, (norm, (ref, std, items)) in self.arms.items():
            if norm == 'per_feature':
                mean, sd = std.mean(0), std.std(0) + 1e-6
                r, s, it = (unit((v - mean) / sd) for v in (ref, std, items))
            elif norm == 'per_image_log':
                r, s, it = (per_image_z(np.log1p(v)) for v in (ref, std, items))
            else:
                r, s, it = (unit(v) for v in (ref, std, items))
            for key, rows in ref_views.items():
                raw = (r[rows] @ it.T).max(0); bg = (r[rows] @ s.T).max(0)
                out[(name, key)] = {'raw': raw, 'z': (raw - bg.mean()) / (bg.std() + 1e-6)}
        hybrid_parts = ('raw_pixels', f'random{RANDOM_SEEDS[0]}|full_concat')
        for key in ref_views:
            z = np.mean([out[(p, key)]['z'] for p in hybrid_parts], 0)
            out[('hybrid_px_random', key)] = {'raw': z, 'z': z}
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('select', 'test'), required=True)
    parser.add_argument('--data', type=Path, required=True); parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='hypernet_v14/checkpoints (random start)')
    parser.add_argument('--trained', type=Path, required=True, help='folder holding spectro_family and spectro_instance runs')
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(); started = time.time()
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    v01 = json.loads((args.benchmark / 'protocol.json').read_text())
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv')
    by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    classes, rare, k = v01['target_classes'], v01['rare_classes'], v01['references_per_class']
    h5 = args.data / 'trainingsetv1d1.h5'

    # Blank detection on the 1.0 s view for every row.
    all_rows = meta.index.to_numpy()
    blank = {}
    for i, im in zip(all_rows, gs.load_split_images(h5, meta.loc[all_rows], '1.0')):
        blank[i] = bool(im.max() == im.min())
    print(f'blank spectrograms: {sum(blank.values())} of {len(blank)} at {time.time() - started:.0f}s', flush=True)

    # References: v0.1 ids, with blanks replaced by a seeded draw from the same class's train pool.
    refs, replaced = {}, []
    for ci, c in enumerate(classes):
        rows = [by_id[g] for g in v01['reference_gravityspy_ids'][c]]
        for r, i in enumerate(rows):
            if blank[i]:
                pool = [j for j in meta.index[(meta['label'] == c) & (meta['sample_type'] == 'train')] if j not in rows and not blank[j]]
                pick = int(np.random.default_rng(20261103 + 1000 * r + ci).permutation(pool)[0])
                replaced.append({'class': c, 'reference': r, 'blank_id': meta.loc[i, 'gravityspy_id'], 'new_id': meta.loc[pick, 'gravityspy_id']})
                rows[r] = pick
        refs[c] = rows
    ref_rows = {i for rows in refs.values() for i in rows}
    pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in ref_rows and not blank[i]])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))

    if sha256(args.runs / RANDOM_CHECKPOINT[0]) != RANDOM_CHECKPOINT[1]:
        raise ValueError('Random start differs from the v0.14 release')
    published = PartDetector(args.runs / RANDOM_CHECKPOINT[0]).model
    models, provenance = {}, {'random_seeds': list(RANDOM_SEEDS)}
    for seed in RANDOM_SEEDS:
        models[f'random{seed}'] = random_encoder(seed)
    shared = {k2: v for k2, v in published.state_dict().items() if not k2.startswith('program_head')}
    if any(not torch.equal(shared[k2], v) for k2, v in models[f'random{RANDOM_SEEDS[0]}'].state_dict().items()):
        raise RuntimeError('Seed-1403 random encoder does not match the published random control')
    for name in TRAINED:
        plan = json.loads((args.trained / name / 'training_plan.json').read_text())
        if plan['spectrograms_used_in_training'] != 0 or plan['borrowed_weights']:
            raise ValueError(f'{name} violates the data budget')
        for e in EPOCHS:
            path = args.trained / name / f'epoch_{e:02d}.pt'
            models[f'{name}_e{e}'] = PartDetector(path).model; provenance[f'{name}_e{e}'] = sha256(path)
        provenance[f'{name}_plan'] = {key: plan[key] for key in ('objective', 'positives', 'loss', 'steps', 'images_per_step', 'curriculum_sha256', 'training_script_sha256')}
    scat = scattering()

    def crops_for(rows, duration):
        return [crop_image(im)[0] for im in gs.load_split_images(h5, meta.loc[rows], duration)]

    ref_list_rows, ref_views = [], {}
    for c, rr in refs.items():
        for r, i in enumerate(rr):
            ref_views[(c, r)] = [len(ref_list_rows) * 2, len(ref_list_rows) * 2 + 1]; ref_list_rows.append(i)

    def all_scores(item_rows):
        per_duration = {}
        for d in gs.DURATIONS:
            refs_c = []
            for crop in crops_for(ref_list_rows, d):
                refs_c += [crop, np.ascontiguousarray(crop[:, ::-1])]
            bank = Bank(models, scat, refs_c, crops_for(std_rows, d), crops_for(item_rows, d))
            per_duration[d] = bank.scores(ref_views)
            print(f'  scored {d}s for {len(item_rows)} items at {time.time() - started:.0f}s', flush=True)
        names = sorted({n for (n, _) in per_duration['1.0']})
        out = {}
        for n in names:
            out[f'{n}|1.0s'] = {key: per_duration['1.0'][(n, key)]['raw'] for key in ref_views}
            out[f'{n}|fused4'] = {key: np.mean([per_duration[d][(n, key)]['z'] for d in gs.DURATIONS], 0) for key in ref_views}
        return out

    dev_rows = np.array([i for i in meta.index[meta['sample_type'] == v01['dev_split']] if not blank[i]])
    dev_labels = meta.loc[dev_rows, 'label'].to_numpy()
    dev_scores = all_scores(dev_rows)
    dev_ap = {n: gs.dev_mean_ap(s, dev_labels, np.ones(len(dev_rows), bool), classes, k) for n, s in dev_scores.items()}
    arm = lambda prefix: [n for n in dev_ap if n.startswith(prefix)]
    best = lambda names: max(names, key=lambda n: dev_ap[n])

    if args.stage == 'select':
        if (args.output / 'protocol.json').exists():
            raise FileExistsError('Selection already frozen')
        selected = {'best_trained': best(arm('spectro_')), 'best_family': best(arm('spectro_family')), 'best_instance': best(arm('spectro_instance')),
                    'best_random': best(arm('random')), 'best_scattering': best(arm('scattering')),
                    'best_raw_pixels': best(arm('raw_pixels')), 'best_hybrid': best(arm('hybrid'))}
        readout = selected['best_random'].split('|', 1)[1]
        seed_spread = {f'random{s}|{readout}': dev_ap[f'random{s}|{readout}'] for s in RANDOM_SEEDS}
        protocol = {'format': 'hypernet.gravity_spy_oneshot.v0.4', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'test_scores_computed_before_freeze': False,
                    'question': 'Do features trained on the spectrogram-like curriculum beat the same architecture at random weights?',
                    'selection_rule': 'Within each arm, the readout x scoring x epoch (or seed) with the highest DEV mean AP',
                    'primary_contrast': 'best_trained minus best_random, TEST mean AP, paired class bootstrap',
                    'secondary_contrasts': ['best_family minus best_instance (objective effect, same data)',
                                            'best_trained minus every random seed at the selected random readout (spread of the control)',
                                            'best_trained minus best_raw_pixels', 'best_trained minus best_scattering', 'best_trained minus best_hybrid',
                                            'best_random minus v0.3 random four-layer 1.0 s (0.576 TEST, blanks included there)'],
                    'selected': selected, 'random_seed_dev_spread': seed_spread, 'dev_mean_ap': dev_ap, 'models': provenance,
                    'blank_rule': BLANK_RULE, 'blank_count_total': int(sum(blank.values())),
                    'dev_items_after_blank_exclusion': int(len(dev_rows)), 'replaced_blank_references': replaced,
                    'reference_gravityspy_ids': {c: meta.loc[rows, 'gravityspy_id'].tolist() for c, rows in refs.items()},
                    'normalization_per_arm': {'cnn': 'per-feature standardization on 600 unlabeled train images', 'raw_pixels': 'per-image z (v0.1)',
                                              'scattering': 'per-feature, or log1p then per-image z; both are arms'},
                    'candidate_epochs': list(EPOCHS), 'readouts': list(READOUTS), 'scorings': list(SCORINGS),
                    'fused4': 'mean over 0.5/1/2/4 s of per-reference z-scores; z statistics from the 600 unlabeled train spectrograms',
                    'scattering_control': 'Kymatio Scattering2D J=3 L=8 on the 96px grayscale crop, untrained',
                    'hybrid_control': 'mean of per-reference z-scores of raw pixels and random1403 four-layer mix',
                    'v01_protocol_sha256': sha256(args.benchmark / 'protocol.json'),
                    'test_reuse_note': 'Fourth use of the v1.1 TEST split (after v0.1, v0.2, v0.3), each under a frozen protocol.',
                    'unlabeled_standardization_images': STANDARDIZE_IMAGES, 'weight_updates_on_spectrograms': 0,
                    'dev_studies': 'benchmarks/gravity_spy_v0.4/dev_studies'}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        print(json.dumps({'selected': selected, 'random_seed_dev_spread': seed_spread, 'replaced_blank_references': replaced,
                          'top_dev': {n: round(v, 4) for n, v in sorted(dev_ap.items(), key=lambda x: -x[1])[:30]}}, indent=2))
        return

    protocol_path = args.output / 'protocol.json'
    if (args.output / 'results.json').exists():
        raise FileExistsError('TEST already scored')
    protocol = json.loads(protocol_path.read_text())
    if any(abs(protocol['dev_mean_ap'][n] - dev_ap[n]) > 1e-9 for n in dev_ap):
        raise RuntimeError('DEV scores changed since the freeze')
    test_rows = np.array([i for i in meta.index[meta['sample_type'] == v01['test_split']] if not blank[i]])
    test_scores = all_scores(test_rows)
    labels = np.concatenate([dev_labels, meta.loc[test_rows, 'label'].to_numpy()])
    dev = np.r_[np.ones(len(dev_rows), bool), np.zeros(len(test_rows), bool)]
    summary = {}
    for n in dev_scores:
        joined = {key: np.concatenate([dev_scores[n][key], test_scores[n][key]]) for key in dev_scores[n]}
        summary[n], _ = gs.evaluate_method(joined, labels, dev, ~dev, classes, rare, k, FAR)
    ap = lambda n: [summary[n]['per_class'][c]['ap'] for c in classes]
    sel = protocol['selected']
    v03 = json.loads((ROOT / 'benchmarks/gravity_spy_v0.3/results.json').read_text())['methods']['random_full_concat']
    pairs = {'best_trained_minus_best_random': (ap(sel['best_trained']), ap(sel['best_random'])),
             'best_family_minus_best_instance': (ap(sel['best_family']), ap(sel['best_instance'])),
             'best_trained_minus_best_raw_pixels': (ap(sel['best_trained']), ap(sel['best_raw_pixels'])),
             'best_trained_minus_best_scattering': (ap(sel['best_trained']), ap(sel['best_scattering'])),
             'best_trained_minus_best_hybrid': (ap(sel['best_trained']), ap(sel['best_hybrid'])),
             'best_random_minus_v03_random_full_concat': (ap(sel['best_random']), [v03['per_class'][c]['ap'] for c in classes])}
    readout = sel['best_random'].split('|', 1)[1]
    for s in RANDOM_SEEDS:
        pairs[f'best_trained_minus_random{s}'] = (ap(sel['best_trained']), ap(f'random{s}|{readout}'))
    contrasts = {}
    for name, (a, b) in pairs.items():
        d, ci = gs.paired_class_bootstrap(a, b); contrasts[name] = {'difference': d, 'class_bootstrap_95ci': ci}
    result = {'format': protocol['format'], 'protocol_sha256': sha256(protocol_path), 'selected': sel,
              'test_items': int(len(test_rows)), 'random_seed_test_mean_ap': {f'random{s}': summary[f'random{s}|{readout}']['mean_ap'] for s in RANDOM_SEEDS},
              'methods': summary, 'contrasts': contrasts, 'runtime_seconds': round(time.time() - started, 1)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    short = lambda s: {key: round(v, 4) for key, v in s.items() if isinstance(v, float)}
    print(json.dumps({'selected': sel, 'selected_test': {r: short(summary[n]) for r, n in sel.items()},
                      'random_seed_test_mean_ap': result['random_seed_test_mean_ap'], 'contrasts': contrasts}, indent=2))


if __name__ == '__main__':
    main()
