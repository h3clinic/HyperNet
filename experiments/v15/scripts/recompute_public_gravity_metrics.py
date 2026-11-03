"""Recompute six frozen GravitySpy arms from public pixel-free score arrays."""
from pathlib import Path
import argparse, json, math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def equal(value, expected, context):
    if not math.isclose(float(value), float(expected), rel_tol=0, abs_tol=1e-12):
        raise ValueError('Published metric differs: ' + context)


def recompute(saved, published):
    labels = saved['test_labels']
    targets = saved['target_labels'].tolist()
    arms = saved['arm_names'].tolist()
    points = saved['operating_points'].tolist()
    groups, inverse = np.unique(saved['test_group_ids'], return_inverse=True)
    truth = labels[:, None] == np.asarray(targets)[None]
    if saved['test_scores'].shape != (len(arms), len(labels), len(targets)):
        raise ValueError('Unexpected compact score dimensions')
    result = {}
    for a, arm in enumerate(arms):
        scores = saved['test_scores'][a]
        reported = published['arms'][arm]['reported_metrics']
        result[arm] = {}
        for p, point in enumerate(points):
            thresholds = saved['thresholds'][a, p]
            alarms = scores >= thresholds[None]
            recall = []
            false_rates = []
            aucs = []
            aps = []
            for t, target in enumerate(targets):
                y = truth[:, t]
                calibration_negative = saved['calibration_scores'][a, saved['calibration_labels'] != target, t]
                budget = 10. if point == 'budget_10_per10000' else 0.
                allowed = math.floor(len(calibration_negative) * budget / 10000)
                threshold = math.nextafter(sorted(calibration_negative, reverse=True)[allowed], math.inf)
                if threshold != thresholds[t]:
                    raise ValueError('Frozen CALIB threshold differs')
                hit = alarms[:, t]
                positive = int(y.sum())
                negative = int((~y).sum())
                tp = int((hit & y).sum())
                fp = int((hit & ~y).sum())
                auc = float(roc_auc_score(y, scores[:, t]))
                ap = float(average_precision_score(y, scores[:, t]))
                expected = reported['per_target'][target][point]
                if threshold != expected['cutoff']['threshold']:
                    raise ValueError('Published threshold differs')
                values = {'positive_events': positive, 'negative_events': negative,
                          'true_positive': tp, 'false_positive': fp, 'recall': tp / positive,
                          'false_alarms_per_10000_negatives': 10000 * fp / negative,
                          'auroc': auc, 'average_precision': ap}
                for key, value in values.items():
                    equal(value, expected['metrics'][key], arm + '/' + target + '/' + key)
                has_positive = np.bincount(inverse, weights=y, minlength=len(groups)) > 0
                group_alarm = np.zeros(len(groups), dtype=bool)
                np.logical_or.at(group_alarm, inverse, hit)
                negative_groups = ~has_positive
                group_metrics = expected['metrics']['negative_time_groups']
                equal(int(negative_groups.sum()), group_metrics['groups'], 'negative groups')
                equal(int((group_alarm & negative_groups).sum()), group_metrics['groups_with_any_false_alarm'], 'group false alarms')
                recall.append(tp / positive)
                false_rates.append(10000 * fp / negative)
                aucs.append(auc)
                aps.append(ap)
            wrong = (alarms & ~truth).any(1)
            recovered = (alarms & truth).any(1)
            present = truth.any(1)
            group_wrong = np.zeros(len(groups), dtype=bool)
            np.logical_or.at(group_wrong, inverse, wrong)
            union = {'screened_events': len(labels), 'wrong_target_alarm_events': int(wrong.sum()),
                     'target_recovered_events': int(recovered.sum()),
                     'events_containing_an_enrolled_target': int(present.sum()),
                     'background_events': int((~present).sum()),
                     'background_events_with_any_alarm': int(alarms[~present].any(1).sum()),
                     'screened_time_groups': len(groups),
                     'time_groups_with_any_wrong_alarm': int(group_wrong.sum()),
                     'wrong_target_alarm_events_per10000_screened_events': 10000 * float(wrong.mean())}
            for key, value in union.items():
                equal(value, reported['union_events'][point][key], arm + '/union/' + key)
            summary = {'macro_recall': float(np.mean(recall)),
                       'macro_false_alarms_per10000_negatives': float(np.mean(false_rates)),
                       'macro_auroc': float(np.mean(aucs)), 'macro_average_precision': float(np.mean(aps)),
                       'union_events': union}
            if point == 'budget_10_per10000':
                for key, saved_key in [('macro_recall', 'recall'),
                                       ('macro_false_alarms_per10000_negatives', 'false_alarms_per10000'),
                                       ('macro_auroc', 'auroc')]:
                    equal(summary[key], reported['macro'][saved_key], arm + '/macro/' + key)
            result[arm][point] = summary
    return result


def bootstrap(saved, expected, plan):
    from analyze_gravity_spy_v15 import resample
    arms = saved['arm_names'].tolist()
    points = saved['operating_points'].tolist()
    targets = saved['target_labels']
    groups, inverse = np.unique(saved['test_group_ids'], return_inverse=True)
    truth = saved['test_labels'][:, None] == targets[None]
    accepted = saved['test_scores'][:, :, None] >= saved['thresholds'][:, None]
    gp = np.zeros((len(groups), len(targets)))
    np.add.at(gp, inverse, truth)
    ge = np.bincount(inverse, minlength=len(groups)).astype(float)
    shape = (len(groups), len(arms), len(points), len(targets))
    tp = np.zeros(shape)
    fp = np.zeros(shape)
    np.add.at(tp, inverse, (accepted & truth[None, :, None]).transpose(1, 0, 2, 3))
    np.add.at(fp, inverse, (accepted & ~truth[None, :, None]).transpose(1, 0, 2, 3))
    samples, counts = resample(gp, ge, tp, fp, draws=plan['valid_draws'], seed=plan['seed'])
    for key, value in counts.items():
        equal(value, expected[key], 'bootstrap/' + key)
    primary = arms.index('primary_final')
    for a, arm in enumerate(arms):
        for p, point in enumerate(points):
            for m, metric in enumerate(plan['metrics']):
                ci = np.quantile(samples[:, a, p, m], [.025, .975], method='linear')
                if not np.allclose(ci, expected['methods'][arm][point][metric]['ci95'], rtol=0, atol=1e-12):
                    raise ValueError('Bootstrap interval differs')
                if a != primary:
                    delta = np.quantile(samples[:, primary, p, m] - samples[:, a, p, m], [.025, .975], method='linear')
                    original = expected['contrasts']['primary_final_minus_' + arm][point][metric]['ci95']
                    if not np.allclose(delta, original, rtol=0, atol=1e-12):
                        raise ValueError('Paired bootstrap contrast differs')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true', help='Also reproduce the 5000 paired time-group bootstrap draws; CPU intensive.')
    args = parser.parse_args()
    folder = ROOT / 'benchmarks/gravity_spy'
    published = json.loads((folder / 'metrics.json').read_text())
    with np.load(folder / 'scores.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    calculated = recompute(saved, published)
    extra = None
    if args.bootstrap:
        base = folder / 'provenance/experiments'
        extra = bootstrap(saved, json.loads((base / 'gravity_spy_paired_analysis.json').read_text()),
                          json.loads((base / 'gravity_spy_bootstrap_plan.json').read_text()))
    print(json.dumps({'matches_published_scores_and_metrics': True, 'real_images_read': 0,
                      'test_events': len(saved['test_event_ids']), 'time_groups': len(np.unique(saved['test_group_ids'])),
                      'methods': calculated, 'bootstrap_recomputed': extra}, indent=2))


if __name__ == '__main__':
    main()
