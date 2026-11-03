"""Predeclared paired time-group uncertainty; no images or model calls."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hypernet.gravity_spy_admission import verify_test_admission
from hypernet.gravity_spy_images import GravitySpyCohort


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample(group_positive, group_events, group_tp, group_fp, draws=5000, seed=1515):
    """Resample shared groups, retaining fixed targets and the first valid draws."""
    groups, arms, points, targets = group_tp.shape
    rng = np.random.default_rng(seed)
    probability = np.full(groups, 1.0 / groups)
    probability /= probability.sum()
    kept = []; accepted = attempted = generated = rejected = 0
    while accepted < draws:
        weights = rng.multinomial(groups, probability, size=128).astype(np.float64)
        generated += len(weights)
        positive = weights @ group_positive
        negative = (weights @ group_events)[:, None] - positive
        valid = (positive > 0).all(1) & (negative > 0).all(1)
        valid_indices = np.flatnonzero(valid)
        needed = draws - accepted
        use = valid_indices[:needed]
        last = int(use[-1]) + 1 if len(use) == needed else len(weights)
        attempted += last
        rejected += int((~valid[:last]).sum())
        if not len(use):
            continue
        tp = (weights[use] @ group_tp.reshape(groups, -1)).reshape(-1, arms, points, targets)
        fp = (weights[use] @ group_fp.reshape(groups, -1)).reshape(-1, arms, points, targets)
        recall = (tp / positive[use, None, None, :]).mean(-1)
        fpr = 10000.0 * (fp / negative[use, None, None, :]).mean(-1)
        kept.append(np.stack([recall, fpr], axis=-1))
        accepted += len(use)
    return np.concatenate(kept), {
        'valid_draws': accepted, 'attempted_draws': attempted,
        'rejected_absent_positive_or_negative_draws': rejected,
        'generated_draws': generated, 'unused_generated_draws_in_final_batch': generated - attempted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/gravity_spy_paired_analysis.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve completed analyses')
    plan_path = ROOT/'experiments/gravity_spy_bootstrap_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    protocol_path = ROOT/'experiments/gravity_spy_protocol_freeze.json'
    if digest(protocol_path) != plan['protocol_sha256']:
        raise ValueError('Analysis plan/protocol binding differs')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    admission = ROOT/'experiments/gravity_spy_test_admission.json'
    verify_test_admission(ROOT/'data/gravity_spy_v15_final', admission)
    cohort = GravitySpyCohort(ROOT/'data/gravity_spy_v15_final', admission)
    expected = cohort.query_events('test')
    event_ids = [row['event_id'] for row in expected]
    groups = sorted({row['group_id'] for row in expected})
    group_index = {name: i for i, name in enumerate(groups)}
    gi = np.array([group_index[row['group_id']] for row in expected])
    targets = plan['target_labels']; arms = list(plan['selected_arms']); points = plan['points']
    if plan['selected_arms'] != protocol['reported_arms']:
        raise ValueError('Reported arm selection changed')
    labels = np.array([row['label'] for row in expected])
    present = labels[:, None] == np.array(targets)[None, :]
    score_sets = {}; inputs = {}
    for model in protocol['models']:
        folder = ROOT/'experiments'/('gravity_test_' + model)
        result_path = folder/'results.json'; score_path = folder/'event_scores.jsonl'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if not result['completed'] or result['mode'] != 'test' or result['scores_sha256'] != digest(score_path):
            raise ValueError('Incomplete or changed TEST outputs: ' + model)
        rows = [json.loads(line) for line in score_path.read_text(encoding='utf-8').splitlines()]
        if [row['event_id'] for row in rows] != event_ids:
            raise ValueError('Different effective TEST roster: ' + model)
        if any(row['label'] != ref['label'] or row['group_id'] != ref['group_id'] for row, ref in zip(rows, expected)):
            raise ValueError('Score-log label/group mismatch')
        score_sets[model] = (rows, result)
        inputs[result_path.relative_to(ROOT).as_posix()] = digest(result_path)
        inputs[score_path.relative_to(ROOT).as_posix()] = digest(score_path)
    accepted = np.zeros((len(arms), len(expected), len(points), len(targets)), dtype=bool)
    for i, arm_name in enumerate(arms):
        arm = plan['selected_arms'][arm_name]; rows, result = score_sets[arm['model']]
        scores = np.array([[row['scores'][arm['method']][target]['score'] for target in targets] for row in rows])
        for j, point in enumerate(points):
            cutoffs = np.array([arm['thresholds'][target][point] for target in targets])
            actual = np.array([result['methods'][arm['method']]['per_target'][target][point]['cutoff']['threshold'] for target in targets])
            if not np.array_equal(cutoffs, actual):
                raise ValueError('TEST cutoffs differ from declared thresholds')
            accepted[i, :, j, :] = scores >= cutoffs
    group_positive = np.zeros((len(groups), len(targets)))
    group_events = np.bincount(gi, minlength=len(groups)).astype(float)
    np.add.at(group_positive, gi, present)
    shape = (len(groups), len(arms), len(points), len(targets))
    group_tp = np.zeros(shape); group_fp = np.zeros(shape)
    np.add.at(group_tp, gi, (accepted & present[None, :, None, :]).transpose(1, 0, 2, 3))
    np.add.at(group_fp, gi, (accepted & ~present[None, :, None, :]).transpose(1, 0, 2, 3))
    positive = group_positive.sum(0); negative = len(expected) - positive
    if (positive == 0).any() or (negative == 0).any():
        raise ValueError('Fixed target set lacks observed positive or negative outcomes')
    estimate = np.stack([(group_tp.sum(0) / positive).mean(-1),
                         10000 * (group_fp.sum(0) / negative).mean(-1)], axis=-1)
    samples, counts = resample(group_positive, group_events, group_tp, group_fp,
                              draws=plan['valid_draws'], seed=plan['seed'])
    method_results = {}; contrasts = {}
    for i, arm in enumerate(arms):
        method_results[arm] = {}
        for j, point in enumerate(points):
            method_results[arm][point] = {}
            for k, metric in enumerate(plan['metrics']):
                method_results[arm][point][metric] = {
                    'estimate': float(estimate[i, j, k]),
                    'ci95': np.quantile(samples[:, i, j, k], [.025, .975], method='linear').tolist()}
    primary_index = arms.index('primary_final')
    for i, arm in enumerate(arms):
        if i == primary_index:
            continue
        key = 'primary_final_minus_' + arm; contrasts[key] = {}
        for j, point in enumerate(points):
            contrasts[key][point] = {}
            for k, metric in enumerate(plan['metrics']):
                contrasts[key][point][metric] = {
                    'difference': float(estimate[primary_index, j, k] - estimate[i, j, k]),
                    'ci95': np.quantile(samples[:, primary_index, j, k] - samples[:, i, j, k],
                                        [.025, .975], method='linear').tolist()}
    result = {'format': 'hypernet.gravity_spy.paired_time_group_analysis.v15',
              'analysis_plan_sha256': digest(plan_path), 'protocol_sha256': digest(protocol_path),
              'analysis_source_sha256': digest(__file__), 'inputs': inputs,
              'numpy_version': np.__version__,
              'blind_test_quality_sha256': digest(ROOT/'data/gravity_spy_v15_final/quality_test_audit.json'),
              'sorted_group_ids_sha256': hashlib.sha256(json.dumps(groups, separators=(',', ':')).encode()).hexdigest(),
              'effective_test_events': len(expected), 'time_groups': len(groups),
              'positive_event_counts': dict(zip(targets, positive.astype(int).tolist())),
              'seed': plan['seed'], **counts, 'methods': method_results, 'contrasts': contrasts,
              'limitations': plan['limitations'] + [
                  'Intervals are pointwise percentile intervals, without familywise multiplicity correction.',
                  'The10-per10,000 budget is empirical CALIB control; observed TEST false-alarm rates can exceed it.'],
              'units': {'macro_recall': 'fraction; multiply by100 for percentage or percentage-point contrasts',
                        'macro_false_alarms_per10000_negatives': 'normalized rate, not10,000 observed negatives'}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output), **counts,
                      'primary': method_results['primary_final']}, indent=2))


if __name__ == '__main__':
    main()
