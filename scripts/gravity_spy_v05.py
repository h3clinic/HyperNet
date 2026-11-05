"""Gravity Spy v0.5: does the v0.4 contrast hold on O3b events no earlier version has seen?

Nothing is selected here. The arms are the ones v0.4 froze on DEV
(`benchmarks/gravity_spy_v0.4/protocol.json` -> `selected`), with the same
references (v1.1 train, HDF5 arrays), the same 600 unlabeled v1.1 train images
for feature standardization and z-statistics, and the same scoring code path.

New in v0.5:
  * haystack: O3b events from Zenodo 5649212, PNG-derived arrays that reproduce
    the HDF5 conversion exactly, one event per 4 s time group;
  * per-reference 1% false-alarm cutoffs calibrated on a disjoint O3b CALIB set;
  * labels are the Gravity Spy CNN's ml_label at confidence >= 0.95 (machine
    labels from a model trained on v1.1; disclosed, not assumed random);
  * the evaluated class set, the verdict rule, all input hashes and the model
    identities are written to protocol.json before TEST is read.

What is fresh: the O3b haystack and CALIB cutoffs. What is not: the references,
the standardization background and the label model all descend from v1.1.

    python scripts/gravity_spy_v05.py --stage freeze ...   # reads CALIB arrays and TEST ids only
    python scripts/gravity_spy_v05.py --stage test ...     # scores O3b TEST once
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
from gravity_spy_benchmark import sha256
from gravity_spy_diagnose import STANDARDIZE_IMAGES, STANDARDIZE_SEED
import gravity_spy_v04 as v4

ARMS = ('best_trained', 'best_instance', 'best_random', 'best_scattering', 'best_raw_pixels', 'best_hybrid')
FAR = 0.01
MIN_CALIB = 5       # a class needs this many CALIB positives to be evaluated
MIN_TEST = 20       # and this many TEST positives
RARE_O3_LIMIT = 30  # evaluated classes with fewer TEST positives are "rare" in the O3b sense
CLASS_BOOTSTRAP = {'draws': 10000, 'seed': 0}
ITEM_BOOTSTRAP = {'draws': 2000, 'seed': 1515}
VERDICT_RULE = ('Confirmed on O3b if best_trained minus best_random in TEST mean AP over evaluated_classes is > 0 and '
                'its paired class-bootstrap 95% interval (10,000 draws, seed 0) excludes zero. Same sign with an '
                'interval that includes zero is reported as "same direction, not confirmed". A difference <= 0 is '
                'reported as "not confirmed". The item bootstrap is a robustness check; the other contrasts are descriptive.')
O3_ONLY_MASK = {'Scattered_Light': 'Fast_Scattering', 'Blip': 'Blip_Low_Frequency'}  # secondary analysis only


def parse_arm(name):
    parts = name.split('|')
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else (parts[0], None, parts[1])


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def scores_sha(scores, ref_views):
    h = hashlib.sha256()
    for arm in ARMS:
        for rk in sorted(ref_views):
            h.update(np.ascontiguousarray(scores[arm][rk], dtype=np.float64).tobytes())
    return h.hexdigest()


def mean_ap_over(scores, labels, classes, k):
    return float(np.mean([np.mean([gs.average_precision(scores[(c, r)], labels == c) for r in range(k)]) for c in classes]))


def load_split(o3, split, roster_df):
    """Arrays with blanks removed; returns dict of arrays, labels, ml_labels, and the exclusion record."""
    a = np.load(o3 / f'{split}_arrays.npz')
    ids = a['gravityspy_id']
    missing = sorted(set(ids) - set(roster_df.index))
    if missing:
        raise RuntimeError(f'{split}: {len(missing)} npz ids not in the roster')
    d1 = a['d1.0']; keep = (d1.max((1, 2)) - d1.min((1, 2))) > 1e-6
    arrays = {'gravityspy_id': ids[keep], **{f'd{d}': a[f'd{d}'][keep] for d in gs.DURATIONS}}
    record = {'in_npz': int(len(ids)), 'blank_excluded': int((~keep).sum()), 'blank_ids': ids[~keep].tolist(), 'scored': int(keep.sum())}
    return arrays, roster_df.loc[arrays['gravityspy_id'], 'target'].to_numpy(), roster_df.loc[arrays['gravityspy_id'], 'ml_label'].to_numpy(), record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('freeze', 'test'), required=True)
    parser.add_argument('--data', type=Path, required=True, help='v1.1 folder (h5 + metadata)')
    parser.add_argument('--o3', type=Path, required=True, help='cohort folder: roster.json, *_roster.csv, *_arrays.npz, *_arrays.json')
    parser.add_argument('--v04', type=Path, default=ROOT / 'benchmarks/gravity_spy_v0.4')
    parser.add_argument('--runs', type=Path, required=True, help='hypernet_v14/checkpoints (published random control)')
    parser.add_argument('--trained', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(); started = time.time()
    import pandas as pd
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    v04 = json.loads((args.v04 / 'protocol.json').read_text()); v04r = json.loads((args.v04 / 'results.json').read_text())
    v01 = json.loads((ROOT / 'benchmarks/gravity_spy_v0.1/protocol.json').read_text())
    classes, k = v01['target_classes'], v01['references_per_class']
    roster = json.loads((args.o3 / 'roster.json').read_text())
    calib_roster = pd.read_csv(args.o3 / 'calib_roster.csv').set_index('gravityspy_id')
    test_roster = pd.read_csv(args.o3 / 'test_roster.csv').set_index('gravityspy_id')

    # Inputs pinned by hash. Both npz files must exist at freeze; TEST arrays are not read at freeze.
    inputs = {name: sha256(args.o3 / name) for name in ('roster.json', 'calib_roster.csv', 'test_roster.csv', 'calib_arrays.npz', 'test_arrays.npz', 'calib_arrays.json', 'test_arrays.json')}
    inputs['resample'] = 'linear'; inputs['script_sha256'] = sha256(Path(__file__)); inputs['v04_script_sha256'] = sha256(ROOT / 'scripts/gravity_spy_v04.py')
    inputs['torch'] = torch.__version__

    # Evaluated classes and label checks, from rosters and npz ids only (no TEST pixels, no scores).
    calib, calib_labels, calib_ml, calib_record = load_split(args.o3, 'calib', calib_roster)
    test_ids_all = np.load(args.o3 / 'test_arrays.npz')['gravityspy_id']
    test_labels_prelim = test_roster.loc[test_ids_all, 'target'].to_numpy()
    counts = {'calib_scored': pd.Series(calib_labels).value_counts().to_dict(), 'test_in_npz': pd.Series(test_labels_prelim).value_counts().to_dict()}
    eval_classes = [c for c in classes if counts['calib_scored'].get(c, 0) >= MIN_CALIB and counts['test_in_npz'].get(c, 0) >= MIN_TEST]
    excluded = {c: {'calib': counts['calib_scored'].get(c, 0), 'test': counts['test_in_npz'].get(c, 0)} for c in classes if c not in eval_classes}
    rare_o3 = sorted(c for c in eval_classes if counts['test_in_npz'][c] < RARE_O3_LIMIT)
    for c in eval_classes:
        if (calib_labels != c).sum() == 0:
            raise RuntimeError(f'no CALIB negatives for {c}')
    other_counts = {split: df[df['target'] == 'other']['ml_label'].value_counts().to_dict() for split, df in (('calib', calib_roster), ('test', test_roster))}

    # References and standardization images: exactly v0.4's.
    meta = pd.read_csv(args.data / 'trainingset_v1d1_metadata.csv'); by_id = {g: i for i, g in enumerate(meta['gravityspy_id'])}
    h5 = args.data / 'trainingsetv1d1.h5'
    refs = {c: [by_id[g] for g in ids] for c, ids in v04['reference_gravityspy_ids'].items()}
    ref_rows = {i for rows in refs.values() for i in rows}
    all_rows = meta.index.to_numpy(); blank = {}
    for i, im in zip(all_rows, gs.load_split_images(h5, meta.loc[all_rows], '1.0')):
        blank[i] = bool(im.max() == im.min())
    pool = np.array([i for i in meta.index[meta['sample_type'] == 'train'] if i not in ref_rows and not blank[i]])
    std_rows = np.sort(np.random.default_rng(STANDARDIZE_SEED).choice(pool, STANDARDIZE_IMAGES, replace=False))

    # Models: identity checked against v0.4 and the published random control.
    selected = {a: v04['selected'][a] for a in ARMS}
    controls = {'raw_pixels', 'hybrid_px_random', 'scattering_perfeature', 'scattering_perimage'}
    needed = {parse_arm(n)[0] for n in selected.values()} - controls
    models, provenance = {}, {}
    for m in sorted(needed | {f'random{v4.RANDOM_SEEDS[0]}'}):
        if m.startswith('random'):
            models[m] = v4.random_encoder(int(m[6:])); provenance[m] = {'torch_seed': int(m[6:]), 'state_sha256': state_sha(models[m])}
        else:
            run, epoch = m.rsplit('_e', 1); path = args.trained / run / f'epoch_{int(epoch):02d}.pt'
            models[m] = PartDetector(path).model; provenance[m] = {'checkpoint_sha256': sha256(path)}
            if v04['models'].get(m) != provenance[m]['checkpoint_sha256']:
                raise RuntimeError(f'{m} checkpoint differs from the one v0.4 scored')
    if sha256(args.runs / v4.RANDOM_CHECKPOINT[0]) != v4.RANDOM_CHECKPOINT[1]:
        raise RuntimeError('Published random control checkpoint differs')
    published = PartDetector(args.runs / v4.RANDOM_CHECKPOINT[0]).model
    shared = {k2: v for k2, v in published.state_dict().items() if not k2.startswith('program_head')}
    if any(not torch.equal(shared[k2], v) for k2, v in models[f'random{v4.RANDOM_SEEDS[0]}'].state_dict().items()):
        raise RuntimeError('Seed-1403 random encoder does not match the published random control')
    scat = v4.scattering()

    def crops_h5(rows, d):
        return [crop_image(im)[0] for im in gs.load_split_images(h5, meta.loc[rows], d)]

    def crops_npz(arrays, d):
        return [crop_image(gs.to_uint8_rgb(a))[0] for a in arrays[f'd{d}']]

    ref_list_rows, ref_views = [], {}
    for c, rr in refs.items():
        for r, i in enumerate(rr):
            ref_views[(c, r)] = [len(ref_list_rows) * 2, len(ref_list_rows) * 2 + 1]; ref_list_rows.append(i)

    def scores_for(arrays):
        per_duration = {}
        for d in gs.DURATIONS:
            refs_c = []
            for crop in crops_h5(ref_list_rows, d):
                refs_c += [crop, np.ascontiguousarray(crop[:, ::-1])]
            bank = v4.Bank(models, scat, refs_c, crops_h5(std_rows, d), crops_npz(arrays, d))
            per_duration[d] = bank.scores(ref_views)
            print(f'  scored {d}s for {len(arrays["gravityspy_id"])} items at {time.time() - started:.0f}s', flush=True)
        out = {}
        for arm, name in selected.items():
            model, readout, scoring = parse_arm(name)
            key = model if readout is None else f'{model}|{readout}'
            if scoring == '1.0s':
                out[arm] = {rk: per_duration['1.0'][(key, rk)]['raw'] for rk in ref_views}
            else:
                out[arm] = {rk: np.mean([per_duration[d][(key, rk)]['z'] for d in gs.DURATIONS], 0) for rk in ref_views}
        return out

    calib_scores = scores_for(calib)
    cutoffs = {arm: {f'{c}|{r}': gs.threshold_at_false_alarm_rate(calib_scores[arm][(c, r)][calib_labels != c], FAR)
                     for c in eval_classes for r in range(k)} for arm in ARMS}
    fingerprint = scores_sha(calib_scores, ref_views)

    if args.stage == 'freeze':
        if (args.output / 'protocol.json').exists():
            raise FileExistsError('Protocol already frozen')
        protocol = {'format': 'hypernet.gravity_spy_oneshot.v0.5', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'question': 'Does the v0.4 trained-minus-random contrast hold on O3b events never seen by any HyperNet version?',
                    'test_scores_computed_before_freeze': False, 'selection_in_this_version': 'none; arms are the v0.4 DEV selections',
                    'what_is_fresh': 'O3b haystack events and CALIB cutoffs; references, standardization background and the label model are v1.1-derived',
                    'arms': selected, 'verdict_rule': VERDICT_RULE, 'expected_effect_v04': v04r['contrasts']['best_trained_minus_best_random'],
                    'evaluated_classes': eval_classes, 'excluded_classes': excluded, 'min_positives_rule': {'calib': MIN_CALIB, 'test': MIN_TEST},
                    'rare_classes_o3': rare_o3, 'rare_rule_o3': f'evaluated classes with fewer than {RARE_O3_LIMIT} TEST positives',
                    'class_bootstrap': CLASS_BOOTSTRAP, 'item_bootstrap': {**ITEM_BOOTSTRAP, 'unit': 'TEST event (every event is its own 4 s group)', 'rule': 'mean over evaluated classes present in the draw'},
                    'secondary_masked_analysis': {'rule': 'for each (reference class -> O3-only label) pair, items with that ml_label are removed from the haystack for that class only', 'pairs': O3_ONLY_MASK},
                    'inputs': inputs, 'calib_scores_sha256': fingerprint, 'far': FAR, 'references_per_class': k,
                    'calib': calib_record, 'counts': counts, 'other_ml_label_counts': other_counts,
                    'roster': {kk: roster[kk] for kk in ('label_source', 'time_grouping_seconds', 'one_event_per_group', 'per_class_caps', 'seed', 'test_events', 'calib_events', 'test_counts', 'calib_counts', 'label_counts_all', 'label_counts_confident', 'v11_id_overlap_in_tables')},
                    'v04_protocol_sha256': sha256(args.v04 / 'protocol.json'), 'references': 'v0.4 references, v1.1 train HDF5 arrays',
                    'standardization': f'{STANDARDIZE_IMAGES} unlabeled v1.1 train images, seed {STANDARDIZE_SEED}, as in v0.4',
                    'cutoffs': f'per reference, {FAR:.0%} false alarms on O3b CALIB negatives', 'models': provenance,
                    'blank_rule': 'arrays whose 1.0 s view has max minus min <= 1e-6 are excluded from CALIB and TEST; ids recorded',
                    'truth_caveat': ('TEST and CALIB labels are predictions of the Gravity Spy CNN, which was trained on the same v1.1 set the '
                                     'references and standardization images come from. The >= 0.95 confidence filter keeps that model\'s most '
                                     'prototypical events, which raises absolute AP for every arm; confident label errors are not assumed random. '
                                     'The primary contrast compares two arms under the same labels.'),
                    'cross_version_comparison': ('Absolute mean AP, recall and one-shot accuracy are not comparable with v0.4: O3b TEST is cap-stratified, '
                                                 'negatives are mostly other target classes at equal counts, and cutoffs come from O3b CALIB. Only the '
                                                 'trained-minus-random contrast is juxtaposed with v0.4\'s.')}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        (args.output / 'cutoffs.json').write_text(json.dumps(cutoffs, indent=1))
        print(json.dumps({'evaluated_classes': eval_classes, 'excluded': excluded, 'rare_o3': rare_o3, 'calib': calib_record,
                          'other_ml_label_counts': other_counts, 'calib_scores_sha256': fingerprint}, indent=1))
        return

    protocol = json.loads((args.output / 'protocol.json').read_text())
    if (args.output / 'results.json').exists():
        raise FileExistsError('TEST already scored')
    drift = {kk: v for kk, v in inputs.items() if protocol['inputs'].get(kk) != v}
    if drift:
        raise RuntimeError(f'Inputs changed since the freeze: {drift}')
    if protocol['calib_scores_sha256'] != fingerprint or protocol['evaluated_classes'] != eval_classes:
        raise RuntimeError('CALIB scores or evaluated classes changed since the freeze')

    test, test_labels, test_ml, test_record = load_split(args.o3, 'test', test_roster)
    for c in eval_classes:
        if (test_labels == c).sum() == 0 or (test_labels != c).sum() == 0:
            raise RuntimeError(f'TEST lacks positives or negatives for {c}')
    test_scores = scores_for(test)
    labels = np.concatenate([calib_labels, test_labels]); is_calib = np.r_[np.ones(len(calib_labels), bool), np.zeros(len(test_labels), bool)]
    summary = {}
    for arm in ARMS:
        joined = {rk: np.concatenate([calib_scores[arm][rk], test_scores[arm][rk]]) for rk in ref_views}
        summary[arm], _ = gs.evaluate_method(joined, labels, is_calib, ~is_calib, eval_classes, rare_o3, k, FAR)
    # Descriptive per-class numbers for the thin classes, no cutoff.
    thin = {}
    for c in excluded:
        pos = test_labels == c
        if pos.sum():
            thin[c] = {'test_positives': int(pos.sum()), **{arm: float(np.mean([gs.average_precision(test_scores[arm][(c, r)], pos) for r in range(k)])) for arm in ARMS}}
    ap = lambda a: [summary[a]['per_class'][c]['ap'] for c in eval_classes]
    contrasts = {}
    for other in ('best_random', 'best_instance', 'best_scattering', 'best_raw_pixels', 'best_hybrid'):
        d, ci = gs.paired_class_bootstrap(ap('best_trained'), ap(other), CLASS_BOOTSTRAP['draws'], CLASS_BOOTSTRAP['seed'])
        contrasts[f'best_trained_minus_{other}'] = {'difference': d, 'class_bootstrap_95ci': ci}
    primary = contrasts['best_trained_minus_best_random']
    verdict = ('confirmed' if primary['difference'] > 0 and primary['class_bootstrap_95ci'][0] > 0 else
               'same direction, not confirmed' if primary['difference'] > 0 else 'not confirmed')
    # Item bootstrap: resample TEST events; mean over evaluated classes present in the draw.
    rng = np.random.default_rng(ITEM_BOOTSTRAP['seed']); n = len(test_labels); draws = []
    for _ in range(ITEM_BOOTSTRAP['draws']):
        pick = rng.integers(0, n, n); lab = test_labels[pick]; present = [c for c in eval_classes if (lab == c).any() and (lab != c).any()]
        a = mean_ap_over({rk: v[pick] for rk, v in test_scores['best_trained'].items()}, lab, present, k)
        b = mean_ap_over({rk: v[pick] for rk, v in test_scores['best_random'].items()}, lab, present, k)
        draws.append(a - b)
    draws = np.array(draws)
    contrasts['best_trained_minus_best_random_item_bootstrap'] = {'difference': primary['difference'], 'item_bootstrap_95ci': [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))], **ITEM_BOOTSTRAP}
    # Secondary: mask O3-only split-off labels for their parent class.
    masked = {}
    for parent, o3label in O3_ONLY_MASK.items():
        if parent not in eval_classes:
            continue
        keep = test_ml != o3label; pos = (test_labels == parent)[keep]
        masked[parent] = {'removed_items': int((~keep).sum()), **{arm: float(np.mean([gs.average_precision(test_scores[arm][(parent, r)][keep], pos) for r in range(k)])) for arm in ARMS},
                          'unmasked': {arm: summary[arm]['per_class'][parent]['ap'] for arm in ARMS}}
    result = {'format': protocol['format'], 'protocol_sha256': sha256(args.output / 'protocol.json'), 'arms': selected, 'verdict': verdict, 'verdict_rule': VERDICT_RULE,
              'test': test_record, 'test_items': int(len(test_labels)), 'test_counts': pd.Series(test_labels).value_counts().to_dict(),
              'evaluated_classes': eval_classes, 'methods': summary, 'thin_classes_descriptive': thin, 'contrasts': contrasts,
              'secondary_masked_analysis': masked, 'v04_primary_contrast': v04r['contrasts']['best_trained_minus_best_random'],
              'v04_test_mean_ap_v11_cohort_not_comparable': {a: v04r['methods'][selected[a]]['mean_ap'] for a in ARMS},
              'runtime_seconds': round(time.time() - started, 1)}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))
    short = lambda s: {kk: round(v, 4) for kk, v in s.items() if isinstance(v, float)}
    print(json.dumps({'verdict': verdict, 'test': {a: short(summary[a]) for a in ARMS}, 'contrasts': contrasts, 'thin': thin, 'masked': masked}, indent=2))


if __name__ == '__main__':
    main()
