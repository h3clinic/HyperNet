"""Fixed-cutoff rare-pattern metrics with linear-time group accounting.

Rates per10,000 are normalized rates, not a claim of10,000 independent images.
Every target remains in the declared target set, including unsupported targets.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta

FORMAT = 'hypernet.v16.pattern_metrics.v1'


def binomial_interval(hits, total, confidence=.95):
    if total < 1 or not 0 <= hits <= total or not 0 < confidence < 1:
        raise ValueError('Expected valid nonempty binomial counts and confidence')
    alpha = 1 - confidence
    return [0. if hits == 0 else float(beta.ppf(alpha / 2, hits, total - hits + 1)),
            1. if hits == total else float(beta.ppf(1 - alpha / 2, hits + 1, total - hits))]


def binomial_upper(hits, total, confidence=.95):
    if total < 1 or not 0 <= hits <= total or not 0 < confidence < 1:
        raise ValueError('Expected valid nonempty binomial counts and confidence')
    return 1. if hits == total else float(beta.ppf(confidence, hits + 1, total - hits))


def calibration_cutoff(negative_scores, budget=10):
    values = np.asarray(negative_scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty calibration negatives are required')
    if not math.isfinite(budget) or not 0 <= budget < 10000:
        raise ValueError('Budget must lie in[0,10000)')
    allowed = math.floor(len(values) * budget / 10000)
    threshold = float(np.nextafter(np.sort(values)[::-1][allowed], math.inf))
    if not math.isfinite(threshold):
        raise ValueError('No finite conservative threshold above the boundary score')
    return {'threshold': threshold, 'calibration_negatives': len(values),
            'allowed_false_alarms': allowed,
            'observed_false_alarms': int(np.count_nonzero(values >= threshold)),
            'target_false_alarms_per_10000': float(budget),
            'source': 'empirical_calibration',
            'claim': 'Empirical calibration constraint; TEST rate is not guaranteed.'}


def ranking_metrics(scores, positive):
    """Tie-block AUROC and average precision, sharing one stable sort."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    positives = int(positive.sum()); negatives = len(positive) - positives
    if not positives:
        return None, None
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]; sorted_positive = positive[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order) - 1]
    cumulative_positive = np.cumsum(sorted_positive, dtype=np.int64)[ends]
    cumulative_negative = ends + 1 - cumulative_positive
    block_positive = np.diff(np.r_[0, cumulative_positive])
    block_negative = np.diff(np.r_[0, cumulative_negative])
    average_precision = float(np.sum(cumulative_positive / (ends + 1) * block_positive) / positives)
    auc = None if not negatives else float(np.sum(
        block_positive * (negatives - cumulative_negative + .5 * block_negative)) / (positives * negatives))
    return auc, average_precision


def _strings(values, name, length=None):
    array = np.asarray(values)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(name + ' must be a one-dimensional matching array')
    if array.dtype.kind not in ('U', 'S', 'O') or any(not isinstance(x, (str, np.str_)) or not x for x in array.tolist()):
        raise ValueError(name + ' must contain nonempty strings')
    return array.astype(str)


def _single(scores, positive, threshold, inverse, group_sizes):
    positives = int(positive.sum()); negatives = len(positive) - positives
    accepted = scores >= threshold
    tp = int(np.count_nonzero(accepted & positive)); fp = int(np.count_nonzero(accepted & ~positive))
    auc, ap = ranking_metrics(scores, positive)
    group_positive = np.bincount(inverse, weights=positive, minlength=len(group_sizes))
    group_alarms = np.bincount(inverse, weights=accepted, minlength=len(group_sizes))
    wholly_negative = group_positive == 0
    gn = int(wholly_negative.sum()); gfp = int(np.count_nonzero(wholly_negative & (group_alarms > 0)))
    mixed = int(np.count_nonzero((group_positive > 0) & (group_positive < group_sizes)))
    missing = []
    if not positives: missing.append('no_positive_events')
    if not negatives: missing.append('no_negative_events')
    return {
        'available': not missing, 'unavailable_reasons': missing,
        'positive_events': positives, 'negative_events': negatives,
        'true_positive': tp, 'false_positive': fp, 'false_negative': positives - tp, 'true_negative': negatives - fp,
        'recall': tp / positives if positives else None,
        'precision': tp / (tp + fp) if tp + fp else None,
        'false_positive_rate': fp / negatives if negatives else None,
        'false_alarms_per_10000_negatives': 10000 * fp / negatives if negatives else None,
        'auroc': auc, 'average_precision': ap, 'threshold': float(threshold),
        'independent_event_binomial_reference': {
            'recall_ci95': binomial_interval(tp, positives) if positives else None,
            'false_alarms_per_10000_ci95': [10000 * x for x in binomial_interval(fp, negatives)] if negatives else None,
            'false_alarms_per_10000_upper95': 10000 * binomial_upper(fp, negatives) if negatives else None,
            'assumption': 'Independent Bernoulli events; not a dependence-adjusted guarantee.'},
        'negative_time_groups': {
            'groups': gn, 'groups_with_any_false_alarm': gfp,
            'mixed_label_groups_excluded_from_group_rate': mixed,
            'any_alarm_groups_per_10000_negative_groups': 10000 * gfp / gn if gn else None,
            'any_alarm_groups_per_10000_upper95': 10000 * binomial_upper(gfp, gn) if gn else None,
            'assumption': 'Independent groups; a group alarms if any member alarms. Distinct groups need not be physically independent.'},
        'rate_scope': 'Normalized per10,000 negatives; observed denominator is negative_events.'}


def aggregate(scores, labels, targets, groups, thresholds=None, budget=10):
    """Return JSON-compatible thresholds, per_target, macro and union metrics.

    If thresholds is absent, only negative labels determine each cutoff.
    Otherwise the supplied cutoff is used verbatim; no test recalibration.
    A macro is None when any fixed target lacks its required denominator.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError('scores must be a finite nonempty[N,T] matrix')
    n, t = scores.shape
    labels = _strings(labels, 'labels', n); targets = _strings(targets, 'targets', t)
    groups = _strings(groups, 'groups', n)
    if len(set(targets)) != t:
        raise ValueError('Targets must be unique')
    if not math.isfinite(budget) or not 0 <= budget < 10000:
        raise ValueError('Invalid calibration budget')
    supplied = thresholds is not None
    if supplied:
        thresholds = np.asarray(thresholds, dtype=np.float64)
        if thresholds.shape != (t,) or not np.isfinite(thresholds).all():
            raise ValueError('One finite frozen threshold per target is required')
    else:
        thresholds = np.empty(t, dtype=np.float64)
    group_names, inverse = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(inverse, minlength=len(group_names))
    per_target = {}
    for index, target in enumerate(targets):
        positive = labels == target
        if supplied:
            cutoff = {'threshold': float(thresholds[index]), 'source': 'supplied_frozen_threshold'}
        else:
            cutoff = calibration_cutoff(scores[~positive, index], budget)
            thresholds[index] = cutoff['threshold']
        metrics = _single(scores[:, index], positive, thresholds[index], inverse, group_sizes)
        metrics['cutoff'] = cutoff; per_target[str(target)] = metrics
    missing_positive = [target for target, row in per_target.items() if not row['positive_events']]
    missing_negative = [target for target, row in per_target.items() if not row['negative_events']]
    def mean_complete(key):
        values = [row[key] for row in per_target.values()]
        return None if any(value is None for value in values) else float(np.mean(values))
    macro = {'fixed_target_count': t, 'missing_positive_targets': missing_positive,
             'missing_negative_targets': missing_negative,
             'all_target_classification_metrics_available': not missing_positive and not missing_negative,
             'recall': mean_complete('recall'),
             'false_alarms_per10000': mean_complete('false_alarms_per_10000_negatives'),
             'auroc': mean_complete('auroc'), 'average_precision': mean_complete('average_precision'),
             'scope': 'Equal target weights. A missing required denominator makes the whole corresponding macro unavailable; no target is dropped or filled with zero.'}
    accepted = scores >= thresholds[None, :]
    target_index = {target: index for index, target in enumerate(targets)}
    true_index = np.array([target_index.get(label, -1) for label in labels])
    has_target = true_index >= 0
    recovered = np.zeros(n, dtype=bool)
    recovered[has_target] = accepted[np.flatnonzero(has_target), true_index[has_target]]
    wrong = accepted.sum(1) - recovered > 0
    wrong_groups = np.bincount(inverse, weights=wrong, minlength=len(group_names)) > 0
    false = int(wrong.sum()); known = int(has_target.sum()); tp = int(recovered.sum())
    gfalse = int(wrong_groups.sum()); gtotal = len(group_names)
    union = {'screened_events': n, 'wrong_target_alarm_events': false,
             'wrong_target_alarm_events_per10000_screened_events': 10000 * false / n,
             'wrong_target_alarm_event_binomial_ci95_per10000': [10000 * x for x in binomial_interval(false, n)],
             'events_containing_an_enrolled_target': known, 'target_recovered_events': tp,
             'target_recovery_recall': tp / known if known else None,
             'target_recovery_binomial_ci95': binomial_interval(tp, known) if known else None,
             'background_events': int((~has_target).sum()),
             'background_events_with_any_alarm': int(np.count_nonzero((~has_target) & accepted.any(1))),
             'screened_time_groups': gtotal, 'time_groups_with_any_wrong_alarm': gfalse,
             'wrong_alarm_time_groups_per10000_groups': 10000 * gfalse / gtotal,
             'wrong_alarm_time_group_binomial_upper95_per10000': 10000 * binomial_upper(gfalse, gtotal),
             'definition': 'An event is a false alarm if any wrong target fires, even if the correct target also fires. Count it once, neverT independent events.',
             'uncertainty_scope': 'Binomial references assume independence; paired time-group bootstrap is supplied separately.'}
    return {'format': FORMAT, 'thresholds': thresholds.tolist(), 'per_target': per_target,
            'macro': macro, 'union': union, 'thresholds_fitted_here': not supplied,
            'calibration_budget_per10000': float(budget),
            'events': n, 'time_groups': gtotal, 'target_labels': targets.tolist()}
