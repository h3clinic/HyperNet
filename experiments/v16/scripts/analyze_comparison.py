"""Saved-score-only, paired time-group analysis with pre-TEST hash bindings.

No image loading, model calls, cutoff selection or architecture selection.
Bindings are supplements to the independently frozen model/data protocol.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.evaluation_metrics import aggregate

FORMAT = 'hypernet.v16.paired_comparison.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')


def load_scores(path):
    required = {'scores', 'event_ids', 'labels', 'group_ids', 'targets', 'methods', 'thresholds', 'pixel_sha256'}
    with np.load(path, allow_pickle=False) as data:
        if not required.issubset(data.files):
            raise ValueError('Missing NPZ fields: ' + ','.join(sorted(required - set(data.files))))
        result = {key: data[key].copy() for key in required}
    scores = np.asarray(result['scores'], dtype=np.float64)
    if scores.ndim != 3 or min(scores.shape) == 0 or not np.isfinite(scores).all():
        raise ValueError('Expected finite nonempty scores[M,N,T]')
    m, n, t = scores.shape; result['scores'] = scores
    for key, count in [('methods', m), ('targets', t), ('event_ids', n), ('labels', n), ('group_ids', n), ('pixel_sha256', n)]:
        values = result[key]
        if values.shape != (count,) or values.dtype.kind != 'U' or np.any(values == ''):
            raise ValueError(key + ' must contain matching nonempty Unicode strings')
    for key in ('methods', 'targets', 'event_ids'):
        if len(np.unique(result[key])) != len(result[key]):
            raise ValueError('Duplicate ' + key)
    if any(len(x) != 64 or set(x.lower()) - set('0123456789abcdef') for x in result['pixel_sha256']):
        raise ValueError('Invalid pixel SHA256')
    thresholds = np.asarray(result['thresholds'], dtype=np.float64)
    if thresholds.shape != (m, t) or not np.isfinite(thresholds).all():
        raise ValueError('Expected finite thresholds[M,T]')
    result['thresholds'] = thresholds
    return result


def validate_plan(data, plan):
    if data['methods'].tolist() != plan['methods']:
        raise ValueError('Method roster/order differs from predeclared plan')
    if plan['primary'] not in plan['methods'] or not set(plan['contrasts']).issubset(plan['methods']):
        raise ValueError('Invalid primary/contrasts')


def source_bindings():
    paths = [Path(__file__).resolve(), ROOT / 'hypernet/evaluation_metrics.py']
    return {str(path): sha(path) for path in paths}


def bind_calibration(calibration, test, plan_path):
    if Path(test).exists():
        raise ValueError('TEST score artifact already exists; cannot declare pre-TEST binding')
    plan = read_json(plan_path); data = load_scores(calibration); validate_plan(data, plan)
    for index, method in enumerate(data['methods']):
        row = aggregate(data['scores'][index], data['labels'], data['targets'], data['group_ids'],
                        budget=plan['budget_false_alarms_per10000'])
        if not np.array_equal(data['thresholds'][index], row['thresholds']):
            raise ValueError('Stored CALIB cutoffs differ from declared negative-only rule: ' + method)
    return {'format': FORMAT + '.calibration_binding', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'plan_path': str(Path(plan_path).resolve()), 'plan_sha256': sha(plan_path),
            'calibration_path': str(Path(calibration).resolve()), 'calibration_sha256': sha(calibration),
            'expected_test_path': str(Path(test).resolve()), 'test_score_artifact_absent_at_binding': True,
            'methods': data['methods'].tolist(), 'targets': data['targets'].tolist(),
            'thresholds': data['thresholds'].tolist(), 'thresholds_digest': digest(data['thresholds'].tolist()),
            'source_files': source_bindings(),
            'scope': 'Filesystem absence of this score artifact is recorded; it is not by itself proof that no earlier inference occurred.'}


def _verify_binding(binding, plan_path, calibration=None, test=None):
    if sha(plan_path) != binding['plan_sha256']:
        raise ValueError('Analysis plan hash changed')
    cal_path = Path(calibration or binding['calibration_path'])
    if sha(cal_path) != binding['calibration_sha256']:
        raise ValueError('CALIB score artifact hash changed')
    if test is not None and Path(test).resolve() != Path(binding['expected_test_path']).resolve():
        raise ValueError('Unexpected TEST artifact path')
    for path, expected in binding['source_files'].items():
        if sha(path) != expected:
            raise ValueError('Analysis source changed after binding: ' + path)
    if digest(binding['thresholds']) != binding['thresholds_digest']:
        raise ValueError('Threshold digest mismatch')


def bind_protocol(protocol, plan_path, binding_path):
    binding = read_json(binding_path); _verify_binding(binding, plan_path)
    if Path(binding['expected_test_path']).exists():
        raise ValueError('TEST score artifact already exists; cannot bind protocol before TEST')
    return {'format': FORMAT + '.protocol_binding', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'protocol_path': str(Path(protocol).resolve()), 'protocol_sha256': sha(protocol),
            'plan_sha256': sha(plan_path), 'calibration_binding_sha256': sha(binding_path),
            'test_score_artifact_absent_at_binding': True}


def group_counts(data):
    """One group factorization; bincount is O(N), with no group-by-event scan."""
    groups, inverse = np.unique(data['group_ids'], return_inverse=True)
    g = len(groups); m, n, t = data['scores'].shape
    positive = data['labels'][:, None] == data['targets'][None, :]
    counts = np.bincount(inverse, minlength=g).astype(np.float64)
    p = np.stack([np.bincount(inverse, weights=positive[:, j], minlength=g) for j in range(t)], axis=1)
    tp = np.empty((g, m, t), dtype=np.float64); fp = np.empty_like(tp)
    for i in range(m):
        accepted = data['scores'][i] >= data['thresholds'][i]
        for j in range(t):
            tp[:, i, j] = np.bincount(inverse, weights=accepted[:, j] & positive[:, j], minlength=g)
            fp[:, i, j] = np.bincount(inverse, weights=accepted[:, j] & ~positive[:, j], minlength=g)
    return groups, counts, p, tp, fp


def weighted_metrics(weights, counts, positives, true_positive, false_positive):
    """Return per-replicate[M] macro rates, preserving every member of a group."""
    p = weights @ positives; n = (weights @ counts)[:, None] - p
    valid = np.all((p > 0) & (n > 0), axis=1)
    m, t = true_positive.shape[1:]
    tp = (weights[valid] @ true_positive.reshape(len(counts), -1)).reshape(-1, m, t)
    fp = (weights[valid] @ false_positive.reshape(len(counts), -1)).reshape(-1, m, t)
    return valid, np.mean(tp / p[valid, None, :], axis=2), 10000 * np.mean(fp / n[valid, None, :], axis=2)


def paired_bootstrap(data, plan):
    validate_plan(data, plan)
    groups, counts, p, tp, fp = group_counts(data)
    no_positive = data['targets'][p.sum(0) == 0].tolist()
    no_negative = data['targets'][(counts.sum() - p.sum(0)) == 0].tolist()
    if no_positive or no_negative:
        return {'available': False, 'missing_positive_targets': no_positive, 'missing_negative_targets': no_negative,
                'reason': 'Fixed macro target set has an undefined denominator. No target was dropped or filled with zero.'}
    size = int(plan['valid_bootstrap_draws']); batch = int(plan['bootstrap_batch_size'])
    if size < 1 or batch < 1: raise ValueError('Positive draw/batch counts required')
    rng = np.random.default_rng(plan['bootstrap_seed']); g = len(groups)
    r, f = [], []; kept = attempted = rejected = unused = 0
    while kept < size:
        if attempted > size * 10000:
            raise RuntimeError('Conditional bootstrap acceptance too low; no fallback to a different target set')
        weights = rng.multinomial(g, np.full(g, 1 / g), size=batch).astype(np.float64)
        valid, recall, false_alarm = weighted_metrics(weights, counts, p, tp, fp)
        selected = np.flatnonzero(valid); take = min(size - kept, len(selected))
        consumed = int(selected[take - 1] + 1) if take and kept + take == size else batch
        attempted += consumed; rejected += consumed - int(valid[:consumed].sum()); unused += batch - consumed
        r.append(recall[:take]); f.append(false_alarm[:take]); kept += take
    draws = {'macro_recall': np.concatenate(r), 'macro_false_alarms_per10000_negatives': np.concatenate(f)}
    point_valid, point_r, point_f = weighted_metrics(np.ones((1, g)), counts, p, tp, fp)
    points = {'macro_recall': point_r[0], 'macro_false_alarms_per10000_negatives': point_f[0]}
    methods = data['methods'].tolist(); primary = methods.index(plan['primary'])
    def interval(values):
        return np.quantile(values, plan['quantiles']['probabilities'], method=plan['quantiles']['method']).tolist()
    arm_results = {method: {metric: {'estimate': float(points[metric][i]), 'ci95': interval(values[:, i])}
                           for metric, values in draws.items()} for i, method in enumerate(methods)}
    contrasts = {}
    for control in plan['contrasts']:
        index = methods.index(control)
        contrasts[plan['primary'] + '_minus_' + control] = {
            metric: {'difference': float(points[metric][primary] - points[metric][index]),
                     'ci95': interval(values[:, primary] - values[:, index])} for metric, values in draws.items()}
    return {'available': True, 'method': 'paired time-group conditional percentile bootstrap',
            'time_groups': g, 'events': int(counts.sum()), 'fixed_targets': data['targets'].tolist(),
            'valid_draws': kept, 'attempted_draws': attempted, 'rejected_draws': rejected,
            'generated_but_unused_draws': unused, 'seed': plan['bootstrap_seed'],
            'numpy_version': np.__version__, 'methods': arm_results, 'contrasts': contrasts,
            'units': {'macro_recall': 'fraction (multiply100 for percent)',
                      'macro_false_alarms_per10000_negatives': 'mean false alarms per10000 target-negative events'},
            'limitations': plan['limitations']}


def analyze(calibration, test, plan_path, binding_path, protocol_binding_path):
    plan = read_json(plan_path); binding = read_json(binding_path); protocol = read_json(protocol_binding_path)
    _verify_binding(binding, plan_path, calibration, test)
    if sha(binding_path) != protocol['calibration_binding_sha256'] or sha(plan_path) != protocol['plan_sha256']:
        raise ValueError('Protocol supplement does not bind this plan/CALIB binding')
    if sha(protocol['protocol_path']) != protocol['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    cal = load_scores(calibration); data = load_scores(test)
    validate_plan(cal, plan); validate_plan(data, plan)
    for key in ('methods', 'targets', 'thresholds'):
        if not np.array_equal(cal[key], data[key]): raise ValueError('CALIB/TEST differ: ' + key)
    if data['targets'].tolist() != binding['targets'] or digest(data['thresholds'].tolist()) != binding['thresholds_digest']:
        raise ValueError('Bound target/cutoff identity mismatch')
    for key in ('event_ids', 'group_ids', 'pixel_sha256'):
        if np.intersect1d(cal[key], data[key]).size: raise ValueError('CALIB/TEST overlap: ' + key)
    aggregates = {}
    for name, split in [('calibration', cal), ('test', data)]:
        aggregates[name] = {str(method): aggregate(split['scores'][i], split['labels'], split['targets'],
            split['group_ids'], thresholds=split['thresholds'][i], budget=plan['budget_false_alarms_per10000'])
            for i, method in enumerate(split['methods'])}
    return {'format': FORMAT, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'primary': plan['primary'], 'methods': plan['methods'], 'aggregates': aggregates,
            'bootstrap': paired_bootstrap(data, plan),
            'inputs': {str(Path(path).resolve()): sha(path) for path in
                      [calibration, test, plan_path, binding_path, protocol_binding_path, protocol['protocol_path']]},
            'source_files': source_bindings(), 'no_model_or_image_calls': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--bind-calibration', action='store_true'); mode.add_argument('--bind-protocol', action='store_true')
    parser.add_argument('--calibration'); parser.add_argument('--test'); parser.add_argument('--protocol')
    parser.add_argument('--plan', default=str(ROOT / 'experiments/comparison_analysis_plan.json'))
    parser.add_argument('--binding'); parser.add_argument('--protocol-binding'); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.bind_calibration:
        if not args.calibration or not args.test: parser.error('--calibration and --test are required')
        result = bind_calibration(args.calibration, args.test, args.plan)
    elif args.bind_protocol:
        if not args.protocol or not args.binding: parser.error('--protocol and --binding are required')
        result = bind_protocol(args.protocol, args.plan, args.binding)
    else:
        if not all([args.calibration, args.test, args.binding, args.protocol_binding]):
            parser.error('--calibration --test --binding --protocol-binding are required')
        result = analyze(args.calibration, args.test, args.plan, args.binding, args.protocol_binding)
    save_new(args.output, result)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'sha256': sha(args.output)}, indent=2))


if __name__ == '__main__': main()
