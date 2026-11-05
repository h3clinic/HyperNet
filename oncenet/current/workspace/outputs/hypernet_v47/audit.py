"""Independent saved-score audit of the v47 development-only fusion study.

Does not import core.py, evaluate.py, their scoring/calibration/metric functions,
or split.py. No neural inference, threshold selection, or reserved-image access.
The older independent audit supplies FIT ordering and tied-ROC mathematics.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import bisect
import collections
import hashlib
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

R = Path(__file__).resolve().parent
V46 = R.parent / 'hypernet_v46'
V45 = R.parent / 'hypernet_v45'
O = R / 'evaluation'
sys.path.insert(0, str(V45))
from seed_development_audit import independent_fit_indices, independent_references
from validation_audit import roc_metrics

TARGETS = ['Low_Frequency_Burst', 'Low_Frequency_Lines', 'Scattered_Light', 'Blip']
DURATIONS = ['d05', 'd1', 'd2', 'd4']
MATCHERS = ['single1s', 'joint4', 'late_mean4', 'late_max4']
POOLS = ['original', 'expanded']
GROUPINGS = ['pooled', 'detector']
ALPHAS = [.001, .0005]
REFS = dict(zip(TARGETS, ['632xGf8n9x', 'Vengh4mj2t', 'dac0D6ni5R', 'yxumvoPABn']))
FIELDS = ['balanced_accuracy', 'mcc', 'recall', 'fa_per10000', 'auroc', 'pauc_std_01', 'pauc_std_001']
STATE_KEYS = ['weight', 'offset', 'ridge', 'beta', 'mu', 'effective_dimension']
_HASH_CACHE = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        _HASH_CACHE[key] = h.hexdigest()
    return _HASH_CACHE[key]


def bind(files):
    for path, digest in files.items():
        require(sha(path) == digest, 'Changed bound artifact: ' + str(path))


def close(actual, expected, name, atol=2e-10, rtol=2e-10):
    if expected is None:
        require(actual is None, name + ': expected null')
    elif isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': fields differ')
        for key, value in expected.items():
            close(actual[key], value, name + '/' + key, atol, rtol)
    elif isinstance(expected, (list, tuple)) and not all(isinstance(x, (int, float)) for x in expected):
        require(actual == expected, name + ': sequence differs')
    elif isinstance(expected, (str, bool)):
        require(actual == expected, name + ': metadata differs')
    else:
        require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), name + ': numerical mismatch')


def counts(rows, indices):
    indices = list(indices)
    basename = lambda r: r['source_csv'].replace('\\', '/').split('/')[-1]
    grouped = collections.Counter((rows[i]['label'], rows[i]['ifo'], basename(rows[i])) for i in indices)
    return dict(n=len(indices),
                label=dict(sorted(collections.Counter(rows[i]['label'] for i in indices).items())),
                ifo=dict(sorted(collections.Counter(rows[i]['ifo'] for i in indices).items())),
                source_csv=dict(sorted(collections.Counter(basename(rows[i]) for i in indices).items())),
                label_ifo_source=[dict(label=k[0], ifo=k[1], source_csv=k[2], n=n) for k, n in sorted(grouped.items())])


def verify_split(sp, rows):
    bind(sp['source_hashes'])
    require(sp['audits']['passed'] and sp['metadata_only'] and sp['no_new_real_holdout'], 'Invalid split scope')
    require(len(rows) == sp['event_count'] == 8250, 'Population changed')
    require(all(r['index'] == i for i, r in enumerate(rows)), 'Not compact v46 row positions')
    require(len({r['event_id'] for r in rows}) == len(rows), 'Repeated event IDs')
    require({r['ifo'] for r in rows} == {'H1', 'L1'}, 'Unexpected detector')
    refs = independent_references(rows, REFS)
    fi = {t: independent_fit_indices(rows, t).tolist() for t in TARGETS}
    union = sorted(set().union(*(set(x) for x in fi.values())))
    used_groups = {rows[i]['group'] for i in union}
    eligible = collections.defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] == 'fit' and row['group'] not in used_groups:
            eligible[row['group']].append(i)
    strata = collections.defaultdict(list)
    for group, indices in eligible.items():
        key = {(rows[i]['label'], rows[i]['ifo'], rows[i]['source_csv'].replace('\\', '/').split('/')[-1]) for i in indices}
        require(len(key) == 1, 'Group crosses strata')
        strata[next(iter(key))].append(group)
    tune, addition, stratum_rows = [], [], []
    for key, group_ids in sorted(strata.items()):
        ordered = sorted(group_ids, key=lambda g: hashlib.sha256(('v47-unused-fit-split|' + g).encode()).hexdigest())
        n = len(ordered) // 3
        ti = [i for g in ordered[:n] for i in eligible[g]]
        ca = [i for g in ordered[n:] for i in eligible[g]]
        tune.extend(ti)
        addition.extend(ca)
        stratum_rows.append(dict(label=key[0], ifo=key[1], source_csv=key[2], groups=len(group_ids), tune=len(ti), cal_add=len(ca)))
    indices = dict(original_fit_union=union, tune=sorted(tune), cal_add=sorted(addition),
                   original_cal=[i for i, r in enumerate(rows) if r['split'] == 'calibration'],
                   development=[i for i, r in enumerate(rows) if r['split'] == 'development'],
                   references=[i for i, r in enumerate(rows) if r['split'] in ['reference', 'reference_check']])
    require(indices == sp['indices'] and stratum_rows == sp['strata'], 'Independent role allocation differs')
    require({k: len(v) for k, v in indices.items()} == dict(original_fit_union=540, tune=474, cal_add=1049, original_cal=4608, development=1558, references=21), 'Role counts differ')
    owners, group_owners = {}, {}
    for role, ix in indices.items():
        require(len(ix) == len(set(ix)), 'Repeated member within role')
        for i in ix:
            require(i not in owners, 'Index appears in two roles')
            owners[i] = role
            group = rows[i]['group']
            require(group not in group_owners or group_owners[group] == role, 'Group spans roles')
            group_owners[group] = role
        close(sp['counts'][role], counts(rows, ix), 'Role composition')
    require(set(owners) == set(range(len(rows))), 'Roles do not cover all rows')
    minimum_role_gap = float('inf')
    for ifo in ['H1', 'L1']:
        ordered = sorted((float(r['event_time']), i) for i, r in enumerate(rows) if r['ifo'] == ifo)
        for (ta, ia), (tb, ib) in zip(ordered, ordered[1:]):
            if owners[ia] != owners[ib]:
                minimum_role_gap = min(minimum_role_gap, tb - ta)
    require(minimum_role_gap > 8, 'Cross-role same-detector time neighbors')
    close(sp['audits']['minimum_cross_role_same_detector_gap_seconds'], minimum_role_gap, 'Cross-role time gap')
    canonical = read(V46 / 'evaluation/fits.json')
    expanded = sorted(indices['original_cal'] + indices['cal_add'])
    expected_targets = {}
    for target in TARGETS:
        negative = lambda ix: [i for i in ix if rows[i]['label'] != target]
        target_split = dict(fit=fi[target], reference=refs[target], tune=negative(indices['tune']),
                            original_cal=negative(indices['original_cal']), expanded_cal=negative(expanded))
        require(sp['targets'][target] == target_split, 'Target-specific roles changed')
        require(len(fi[target]) == 512 and len(set(fi[target])) == 512, 'FIT512 changed')
        for model_fits in canonical.values():
            for view in ['single', 'multi']:
                old = model_fits[view + '__' + target]
                require(old['fit_indices'] == fi[target] and old['reference_index'] == refs[target] and old['calibration_indices'] == target_split['original_cal'], 'Frozen v46 membership changed')
        for key in ['fit', 'tune', 'original_cal', 'expanded_cal']:
            close(sp['target_counts'][target][key], counts(rows, target_split[key]), 'Target role composition')
        expected_targets[target] = target_split
    require(sp['reference_ids'] == REFS, 'Reference IDs changed')
    reserve_path = V45 / 'reserve_metadata_audit.json'
    reserve = read(reserve_path)
    reserved = reserve['records']
    require(len(reserved) == reserve['remaining_after_v44_ids_and_eight_seconds'] == 195, 'Reserve metadata changed')
    require(sha(reserve['original_candidate_path']) == reserve['original_candidate_sha256'], 'Reserve source metadata changed')
    require(not {r['event_id'] for r in reserved}.intersection(r['event_id'] for r in rows), 'Reserve event consumed')
    nearest = float('inf')
    for ifo in ['H1', 'L1']:
        times = sorted(float(r['event_time']) for r in reserved if r['ifo'] == ifo)
        for row in (r for r in rows if r['ifo'] == ifo):
            time = float(row['event_time']); at = bisect.bisect_left(times, time)
            for j in [at - 1, at]:
                if 0 <= j < len(times):
                    nearest = min(nearest, abs(times[j] - time))
    require(nearest > 8, 'Reserved event temporal neighbor consumed')
    close(sp['audits']['minimum_reserved_same_detector_gap_seconds'], nearest, 'Reserve time gap')
    return expected_targets, indices, dict(events=len(rows), role_counts={k: len(v) for k, v in indices.items()},
        metadata_strata=len(strata), group_disjoint=True, minimum_cross_role_gap_seconds=minimum_role_gap,
        reserved_metadata_events=len(reserved), minimum_reserved_gap_seconds=nearest, reserve_closed=True,
        no_reserved_images_opened=True, reference_event_ids=REFS)


def verify_normal_equation(feature, fi, ri, state, scores):
    bg = np.asarray(feature[fi], np.float64)
    mu = bg.mean(0); centered = bg - mu
    beta = .5 / (len(fi) - 1)
    ridge = max(.5 * np.square(centered).sum() / ((len(fi) - 1) * feature.shape[1]), 1e-12)
    w = np.asarray(state['weight'], np.float64)
    target = np.asarray(feature[ri], np.float64) - mu
    require(w.shape == target.shape and np.isfinite(w).all(), 'Invalid saved weight')
    residual = ridge * w + beta * (centered.T @ (centered @ w)) - target
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-12))
    require(relative <= 2e-7, 'ELDA normal-equation residual exceeds tolerance')
    close(state['mu'], mu, 'ELDA mean', 2e-12, 2e-12)
    close(state['beta'], beta, 'ELDA beta')
    close(state['ridge'], ridge, 'ELDA ridge', 1e-16, 2e-10)
    require(int(state['effective_dimension']) == feature.shape[1], 'Effective dimension changed')
    close(state['offset'], -mu @ w, 'ELDA intercept', 2e-7, 2e-8)
    maximum = 0.
    for start in range(0, len(feature), 512):
        calculated = np.asarray(feature[start:start + 512], np.float64) @ w + state['offset']
        maximum = max(maximum, float(np.max(np.abs(calculated - scores[start:start + 512]))))
        close(scores[start:start + 512], calculated, 'All raw scores', 2e-7, 2e-8)
    return dict(normal_equation_relative_residual=relative, maximum_raw_score_difference=maximum,
                ridge=ridge, dimensions=feature.shape[1], fit_negatives=len(fi))


def metrics(y, ranking, decision, indices):
    yy = np.asarray(y, bool)[indices]
    vv = np.asarray(ranking, np.float64)[indices]
    pr = np.asarray(decision, bool)[indices]
    positive = int(yy.sum()); negative = len(yy) - positive
    require(positive > 0 and negative > 0, 'Metric stratum lacks either outcome')
    tp = int(np.count_nonzero(yy & pr)); fp = int(np.count_nonzero(~yy & pr))
    tn = negative - fp; fn = positive - tp
    recall = tp / positive; specificity = tn / negative
    product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ranked = roc_metrics(yy, vv, (.01, .001), budgets=())
    return dict(positive=positive, negative=negative, tp=tp, fp=fp, tn=tn, fn=fn,
                balanced_accuracy=(recall + specificity) / 2,
                mcc=(tp * tn - fp * fn) / math.sqrt(product) if product else 0.,
                precision=tp / (tp + fp) if tp + fp else None, recall=recall,
                specificity=specificity, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                mcc_degenerate=not bool(product), fa_per10000=fp / negative * 10000,
                auroc=ranked['auroc'],
                pauc_std_01=ranked['partial_auc']['0.01']['standardized_partial_auc'],
                pauc_std_001=ranked['partial_auc']['0.001']['standardized_partial_auc'])


def calibration_expected(scores, rows, ci, grouping):
    # Independent integer comparison counts rather than the evaluator's searchsorted.
    pv = np.empty(len(rows), np.float64)
    decisions = {a: np.zeros(len(rows), bool) for a in ALPHAS}
    details = {}
    for name in (['all'] if grouping == 'pooled' else ['H1', 'L1']):
        eligible = np.array([i for i, r in enumerate(rows) if name == 'all' or r['ifo'] == name], int)
        selected = np.array([i for i in ci if name == 'all' or rows[i]['ifo'] == name], int)
        require(len(selected) > 0 and len(eligible) > 0, 'Empty calibration group')
        background = np.asarray(scores[selected], np.float64)
        count = np.empty(len(eligible), np.int64)
        for start in range(0, len(eligible), 128):
            q = scores[eligible[start:start + 128]]
            count[start:start + 128] = 1 + np.count_nonzero(background[:, None] >= q[None, :], axis=0)
        pv[eligible] = count / (len(selected) + 1)
        cuts = {}
        ordered = sorted(background.tolist())
        for alpha in ALPHAS:
            fraction = Fraction(str(alpha))
            allowed = ((len(selected) + 1) * fraction.numerator) // fraction.denominator
            cut = ordered[len(selected) - allowed] if allowed >= 1 else None
            decision = count * fraction.denominator <= (len(selected) + 1) * fraction.numerator
            require(np.array_equal(decision, scores[eligible] > (cut if cut is not None else np.inf)), 'Strict cutoff / conservative p-value equivalence')
            decisions[alpha][eligible] = decision
            cuts[str(alpha)] = dict(threshold=cut, rank_budget=allowed, supported=cut is not None)
        details[name] = dict(calibration_indices=selected.tolist(), n=len(selected), minimum_p=1 / (len(selected) + 1), cutoffs=cuts)
    require(np.isfinite(pv).all() and ((pv > 0) & (pv <= 1)).all(), 'Invalid p-values')
    return pv, decisions, details


def table_verify(actual, expected, key_fields, entries):
    require(len(actual) == len(expected), 'Wrong table row count: ' + str(key_fields))
    seen = set(); by_id = {e['id']: e for e in entries}
    for row in actual:
        key = tuple(row[k] for k in key_fields)
        require(key not in seen and key in expected, 'Duplicate/unexpected table cell')
        seen.add(key)
        e = by_id[row['model']]
        require(row['seed'] == e['seed'] and row['training'] == e['training'], 'Model table metadata differs')
        for field, value in expected[key].items():
            close(row[field], value, str(key) + '/' + field)


def audit(protocol_sha256):
    threadpool_limits(2)
    protocol = read(R / 'protocol.json')
    require(sha(R / 'protocol.json') == protocol_sha256, 'Wrong frozen v47 protocol')
    bind(protocol['files'])
    require(protocol['files'].get(str(Path(__file__).resolve())) == sha(__file__), 'Auditor not frozen before scoring')
    require(protocol['matchers'] == MATCHERS and protocol['calibration_pools'] == POOLS and protocol['groupings'] == GROUPINGS and protocol['alphas'] == ALPHAS, 'Candidate set changed')
    require(protocol['execution_rounds'] == GROUPINGS and protocol['all_rounds_declared_before_scores'], 'Sequential adaptive rounds')
    require(protocol['reserve_closed'] and protocol['development_only'] and protocol['one_positive_event'], 'Scope changed')
    prior_protocol = read(V46 / 'evaluation/protocol.json')
    prior_audit = read(V46 / 'evaluation/audit.json')
    require(prior_audit['passed'] and prior_audit['results_sha256'] == sha(V46 / 'evaluation/results.json'), 'Prior result audit mismatch')
    require(prior_audit['protocol_sha256'] == sha(V46 / 'evaluation/protocol.json'), 'Prior protocol binding mismatch')
    bind(prior_protocol['files'])
    for dependency in [V45 / 'seed_development_audit.py', V45 / 'validation_audit.py']:
        require(prior_protocol['files'].get(str(dependency)) == sha(dependency), 'Independent helper not bound')
    require(prior_audit['audit_code_sha256'] == sha(V46 / 'evaluation_audit.py'), 'Prior audit source changed')
    check = read(R / 'checks.json')
    require(check['passed'] and check['core_sha256'] == sha(R / 'core.py'), 'Pre-score controls missing')
    result = read(O / 'results.json')
    require(result['complete'] and result['development_only'] and result['reserve_closed'] and result['no_selection'], 'Incomplete/adaptive result scope')
    require(result['protocol_sha256'] == protocol_sha256, 'Result protocol mismatch')
    bind(result['files'])
    require(read(O / 'status.json')['stage'] == 'complete' and read(O / 'status.json')['complete'], 'Final status not complete')
    started = read(O / 'started.json')
    require(started['protocol_sha256'] == protocol_sha256 and started['epoch'] >= protocol['frozen_epoch'], 'Scoring started before freeze')
    rows = read(V46 / 'data/records.json'); sp = read(R / 'splits.json')
    target_splits, indices, split_details = verify_split(sp, rows)
    reg = read(V46 / 'evaluation/registry.json'); entries = reg['models']
    require(protocol['models'] == entries and len(entries) == 11 and len({e['id'] for e in entries}) == 11, 'Model factorial changed')
    require({(e['seed'], e['training']) for e in entries if e['seed'] is not None} == {(s, t) for s in [45011, 45029, 45047] for t in ['random', 'shapes', 'spectrogram']}, 'Incomplete seed pairing')
    bind(reg['source_registries'])
    labels = np.array([r['label'] for r in rows]); dev = np.array(indices['development'], int)
    expected_rows, expected_detectors, expected_labels, expected_macros, expected_unions = {}, {}, {}, {}, {}
    all_details = read(O / 'calibration.json'); expected_details = {}
    normal_checks, normalization_checks, parity_checks = [], [], []
    for entry in entries:
        model = entry['id']; folder = O / model
        if 'path' in entry:
            require(sha(entry['path']) == entry['sha256'], 'Frozen model weight bytes changed')
        bind(entry.get('files', {}))
        b = read(folder / 'binding.json'); bind(b['files'])
        require(b['protocol_sha256'] == protocol_sha256, 'Score cache protocol mismatch')
        for path in ['raw.npz', 'states.npz', 'fused.npz', 'normalization.json']:
            require(b['files'].get(str(folder / path)) == sha(folder / path), 'Missing score artifact binding')
            require(result['files'].get(str(folder / path)) == sha(folder / path), 'Missing result artifact binding')
        feature_folder = V46 / 'evaluation/features' / model
        fb = read(feature_folder / 'binding.json'); bind(fb['files'])
        require(fb['settings']['row_sha256'] == sha(V46 / 'data/records.json') and fb['settings']['registry_sha256'] == sha(V46 / 'evaluation/registry.json') and fb['settings']['protocol_sha256'] == sha(V46 / 'evaluation/protocol.json'), 'Feature source alignment differs')
        dim = 4032 if entry['kind'] == 'cnn' else 1536
        features = {d: np.load(feature_folder / (d + '.npy'), mmap_mode='r', allow_pickle=False) for d in DURATIONS}
        for d, f in features.items():
            require(f.shape == (len(rows), dim) and f.dtype == np.float32 and np.isfinite(f).all(), 'Feature shape/dtype/finite mismatch')
            close(np.linalg.norm(f, axis=1), 1, 'Unit descriptor', 1e-5, 1e-5)
            require(protocol['files'].get(str(feature_folder / (d + '.npy'))) == sha(feature_folder / (d + '.npy')), 'Feature not frozen')
        with np.load(folder / 'raw.npz', allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        with np.load(folder / 'states.npz', allow_pickle=False) as z:
            states = {k: z[k] for k in z.files}
        with np.load(folder / 'fused.npz', allow_pickle=False) as z:
            fused = {k: z[k] for k in z.files}
        with np.load(V46 / 'evaluation' / (model + '_scores.npz'), allow_pickle=False) as z:
            previous = {k: z[k] for k in z.files}
        normalization = read(folder / 'normalization.json')
        require(set(raw) == set(normalization) == set(TARGETS), 'Raw/normalization target keys')
        require(set(states) == {t + '__' + d + '__' + k for t in TARGETS for d in DURATIONS for k in STATE_KEYS}, 'Raw state completeness')
        require(set(fused) == {m + '__' + t for m in MATCHERS for t in TARGETS}, 'Fused score completeness')
        saved_parity = {x['target']: x for x in b['parity']}
        require(len(saved_parity) == len(b['parity']) == 4, 'Single/joint parity completeness')
        for target in TARGETS:
            ts = target_splits[target]; x = raw[target]
            require(x.shape == (len(rows), 4) and np.isfinite(x).all(), 'Invalid raw score array')
            for j, d in enumerate(DURATIONS):
                state = {k: states[target + '__' + d + '__' + k] for k in STATE_KEYS}
                normal_checks.append(dict(model=model, target=target, duration=d, **verify_normal_equation(features[d], ts['fit'], ts['reference'], state, x[:, j])))
            tune = x[ts['tune']]
            location = np.percentile(tune, 50, axis=0)
            mad = 1.4826 * np.percentile(np.abs(tune - location), 50, axis=0)
            std = np.sqrt(np.square(tune - tune.mean(0)).sum(0) / (len(tune) - 1))
            scales = np.array([mad[i] if mad[i] > 1e-12 else (std[i] if std[i] > 1e-12 else 1.) for i in range(4)])
            fallback = ['mad' if mad[i] > 1e-12 else ('std' if std[i] > 1e-12 else 'unit') for i in range(4)]
            norm = normalization[target]
            close(norm['location'], location, 'TUNE location'); close(norm['scale'], scales, 'TUNE scale')
            require(norm['fallback'] == fallback, 'TUNE fallback differs')
            standardized = (x - location) / scales
            expected_fused = dict(single1s=x[:, 1], joint4=previous['multi__' + target], late_mean4=np.sum(standardized, axis=1) / 4, late_max4=np.maximum.reduce(standardized, axis=1))
            for method, score in expected_fused.items():
                close(fused[method + '__' + target], score, 'Fused score: ' + method)
            close(x[:, 1], previous['single__' + target], 'v46 single-score parity', 2e-7, 2e-8)
            require(np.array_equal(fused['joint4__' + target], previous['multi__' + target]), 'v46 joint score not exact')
            pe = float(np.max(np.abs(x[:, 1] - previous['single__' + target])))
            close(saved_parity[target]['max_single_error'], pe, 'Reported parity error')
            require(saved_parity[target]['original_single_decisions_identical'] and saved_parity[target]['joint_exact'], 'Reported parity flags failed')
            for alpha in ALPHAS:
                a = Fraction(str(alpha)); n = len(ts['original_cal']); k = ((n + 1) * a.numerator) // a.denominator
                require(k >= 1, 'Original calibration unsupported')
                old_cut = sorted(previous['single__' + target][ts['original_cal']])[-k]
                new_cut = sorted(x[ts['original_cal'], 1])[-k]
                require(np.array_equal(previous['single__' + target][dev] > old_cut, x[dev, 1] > new_cut), 'Original single decisions changed')
            normalization_checks.append(dict(model=model, target=target, tune_negatives=len(ts['tune']), fallback=fallback))
            parity_checks.append(dict(model=model, target=target, maximum_single_score_error=pe, joint_bit_exact=True, original_decisions_identical=True))
        for grouping in GROUPINGS:
            with np.load(folder / (grouping + '_pvalues.npz'), allow_pickle=False) as z:
                saved_p = {k: z[k] for k in z.files}
            with np.load(folder / (grouping + '_decisions.npz'), allow_pickle=False) as z:
                saved_d = {k: z[k] for k in z.files}
            p_keys = {pool + '__' + method + '__' + t for pool in POOLS for method in MATCHERS for t in TARGETS}
            require(set(saved_p) == p_keys and set(saved_d) == {key + '__' + str(a) for key in p_keys for a in ALPHAS}, 'Calibration cache incomplete')
            expected_details[model + '__' + grouping] = {}
            for pool in POOLS:
                for method in MATCHERS:
                    target_decisions = {}
                    for target in TARGETS:
                        score = fused[method + '__' + target]
                        ci = target_splits[target]['original_cal' if pool == 'original' else 'expanded_cal']
                        pv, predictions, details = calibration_expected(score, rows, ci, grouping)
                        ck = pool + '__' + method + '__' + target
                        expected_details[model + '__' + grouping][ck] = details
                        close(saved_p[ck], pv, 'All conservative p-values', 0, 0)
                        ranking = score if grouping == 'pooled' else -pv
                        for alpha in ALPHAS:
                            decision = predictions[alpha]
                            require(saved_d[ck + '__' + str(alpha)].dtype == np.bool_ and np.array_equal(saved_d[ck + '__' + str(alpha)], decision), 'Saved decisions differ')
                            target_decisions[(target, alpha)] = decision
                            key = (model, method, pool, grouping, target, alpha)
                            expected_rows[key] = metrics(labels == target, ranking, decision, dev)
                            for ifo in ['H1', 'L1']:
                                ix = np.array([i for i in dev if rows[i]['ifo'] == ifo], int)
                                expected_detectors[key + (ifo,)] = metrics(labels == target, ranking, decision, ix)
                            for label in sorted(set(labels[dev]) - {target}):
                                ix = dev[labels[dev] == label]
                                expected_labels[key + (label,)] = dict(fp=int(decision[ix].sum()), negative=len(ix))
                    for alpha in ALPHAS:
                        key = (model, method, pool, grouping, alpha)
                        target_values = [expected_rows[(model, method, pool, grouping, t, alpha)] for t in TARGETS]
                        expected_macros[key] = {k: float(np.mean([v[k] for v in target_values])) for k in FIELDS} | dict(tp=sum(v['tp'] for v in target_values), fp=sum(v['fp'] for v in target_values))
                        union = np.logical_or.reduce([target_decisions[(t, alpha)] for t in TARGETS])
                        bg = dev[~np.isin(labels[dev], TARGETS)]
                        for label in ['ALL_NON_TARGET'] + sorted(set(labels[bg])):
                            ix = bg if label == 'ALL_NON_TARGET' else bg[labels[bg] == label]
                            expected_unions[key + (label,)] = dict(fp=int(union[ix].sum()), negative=len(ix), fa_per10000=float(union[ix].mean() * 10000))
        print('V47 AUDITED', model, flush=True)
        del features, raw, states, fused, previous
    close(all_details, expected_details, 'All calibration indices/thresholds/groups')
    base_keys = ['model', 'matcher', 'pool', 'grouping']
    tables = [('rows', expected_rows, base_keys + ['target', 'alpha']),
              ('detector_rows', expected_detectors, base_keys + ['target', 'alpha', 'detector']),
              ('false_alarm_by_label', expected_labels, base_keys + ['target', 'alpha', 'negative_label']),
              ('macro', expected_macros, base_keys + ['alpha']),
              ('any_target_false_alarm', expected_unions, base_keys + ['alpha', 'negative_label'])]
    for name, expected, keys in tables:
        table_verify(result[name], expected, keys, entries)
    require(len(expected_rows) == 16 * 11 * 2 * 4 and len(normal_checks) == 16 * 11, 'Factorial/count mismatch')
    for grouping in GROUPINGS:
        checkpoint = read(O / ('round_' + grouping + '.json'))
        require(checkpoint['complete'] and checkpoint['no_setting_changes'] and checkpoint['grouping'] == grouping and checkpoint['protocol_sha256'] == protocol_sha256, 'Round checkpoint mismatch')
        expected = {k: v for k, v in expected_macros.items() if k[3] == grouping}
        table_verify(checkpoint['macro'], expected, base_keys + ['alpha'], entries)
    # Detect source mutation during this audit, including changes after a cached hash.
    bind(protocol['files']); bind(result['files'])
    return dict(passed=True, protocol_sha256=protocol_sha256, results_sha256=sha(O / 'results.json'),
                audit_code_sha256=sha(__file__), prior_audit_sha256=sha(V46 / 'evaluation/audit.json'),
                split_audit=split_details, configurations=16, models=11, target_metric_rows=len(expected_rows),
                normal_equation_checks=normal_checks, normalization_checks=normalization_checks,
                original_control_parity=parity_checks, calibration_groups_verified=sum(len(v) for d in expected_details.values() for v in d.values()),
                table_counts={name: len(expected) for name, expected, _ in tables},
                tied_tail_counts_independently_recomputed=True, conditional_ranking_is_negative_pvalue=True,
                independent_helpers={str(p): sha(p) for p in [V45 / 'seed_development_audit.py', V45 / 'validation_audit.py']},
                no_encoder_inference=True, no_reserved_images_opened=True, reserve_closed=True,
                scope='Exploratory, repeatedly consumed DEV. One positive event and four PNGs plus labeled FIT/TUNE/CAL; no fresh confirmation or certified deployment FPR.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', default=str(O / 'audit.json'))
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    require(R in destination.parents and not destination.exists(), 'Invalid audit output or overwrite')
    result = audit(args.protocol_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['normal_equation_checks', 'normalization_checks', 'original_control_parity']}, indent=2))
