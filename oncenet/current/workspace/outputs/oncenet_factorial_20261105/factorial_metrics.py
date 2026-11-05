"""Frozen-score factorial analysis; no inference, fitting, file I/O or thresholds.

Axis contract: scores[training, matcher, seed, reference_episode, target, event].
Training order is random, trained; matcher order is cosine, negative_aware.
Thresholds have the same first five axes, followed by alpha (.001, .01).
The caller must commit all scores and bind this source before joining labels.
Repeated seeds/references reuse events; they never increase the sample count.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

ALPHAS = (0.001, 0.01)
TRAINING = ("random", "trained")
MATCHERS = ("cosine", "negative_aware")
METRICS = (
    "auroc", "pauc_std_001", "pauc_std_01",
    "recall_empirical_fpr_at_most_001", "recall_empirical_fpr_at_most_01",
    "balanced_accuracy_alpha001", "mcc_alpha001", "recall_alpha001",
    "specificity_alpha001", "fa_per10000_alpha001",
    "balanced_accuracy_alpha01", "mcc_alpha01", "recall_alpha01",
    "specificity_alpha01", "fa_per10000_alpha01",
)
CONTRASTS = (
    "training_main", "matcher_main", "interaction",
    "training_with_cosine", "training_with_negative_aware",
    "matcher_with_random", "matcher_with_trained",
)


def _contrasts(cell: np.ndarray) -> np.ndarray:
    """Cell is [2 training, 2 matcher, metric]. Return paired differences."""
    random_cos, random_neg = cell[0]
    trained_cos, trained_neg = cell[1]
    return np.stack((
        ((trained_cos - random_cos) + (trained_neg - random_neg)) / 2,
        ((random_neg - random_cos) + (trained_neg - trained_cos)) / 2,
        (trained_neg - random_neg) - (trained_cos - random_cos),
        trained_cos - random_cos, trained_neg - random_neg,
        random_neg - random_cos, trained_neg - trained_cos,
    ))


def _partial(fpr: np.ndarray, tpr: np.ndarray, limit: float) -> tuple[float, float]:
    stop = int(np.searchsorted(fpr, limit, side="right"))
    interpolated = float(np.interp(limit, fpr, tpr))
    xx = np.concatenate((fpr[:stop], [limit]))
    yy = np.concatenate((tpr[:stop], [interpolated]))
    area = float(np.sum(np.diff(xx) * (yy[1:] + yy[:-1]) / 2))
    standardized = .5 * (1 + (area - .5 * limit * limit) / (limit - .5 * limit * limit))
    empirical = float(tpr[stop - 1])
    return standardized, empirical


class _Ranking:
    """Cache score order/ties; each bootstrap only changes group multiplicities."""
    def __init__(self, y: np.ndarray, score: np.ndarray,
                 group_index: np.ndarray, thresholds: np.ndarray):
        order = np.argsort(-score, kind="stable")
        self.y = y[order].astype(np.float64)
        self.groups = group_index[order]
        sorted_score = score[order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(score) - 1]
        # Number of observations strictly above each frozen threshold.
        self.cuts = np.searchsorted(-sorted_score, -thresholds, side="left")

    def measure(self, multiplicity: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        weights = multiplicity[self.groups].astype(np.float64, copy=False)
        positive = np.cumsum(weights * self.y)
        negative = np.cumsum(weights * (1. - self.y))
        pos, neg = float(positive[-1]), float(negative[-1])
        if pos <= 0 or neg <= 0:
            raise ValueError("Undefined ROC: resample has no positives or no negatives")
        # Zero-weight tie groups are harmless: they produce repeated ROC points.
        fpr = np.r_[0., negative[self.ends] / neg]
        tpr = np.r_[0., positive[self.ends] / pos]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        p001, e001 = _partial(fpr, tpr, .001)
        p01, e01 = _partial(fpr, tpr, .01)
        values = [auc, p001, p01, e001, e01]
        rows = []
        for alpha, cut in zip(ALPHAS, self.cuts):
            tp = float(positive[cut - 1]) if cut else 0.
            fp = float(negative[cut - 1]) if cut else 0.
            fn, tn = pos - tp, neg - fp
            recall, specificity = tp / pos, tn / neg
            denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            mcc = (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0.
            ba, fa = .5 * (recall + specificity), 10000. * fp / neg
            values.extend((ba, mcc, recall, specificity, fa))
            rows.append(dict(alpha=alpha, positive=int(pos), negative=int(neg),
                             tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                             balanced_accuracy=ba, mcc=mcc, recall=recall,
                             specificity=specificity, fa_per10000=fa))
        return np.asarray(values), rows


def _interval(values: np.ndarray, confidence: float) -> dict | None:
    if len(values) == 0:
        return None
    tail = (1. - confidence) / 2
    lo, hi = np.quantile(values, (tail, 1. - tail))
    return dict(low=float(lo), high=float(hi), confidence=confidence)


def _binomial_interval(fp: int, negative: int) -> list[float] | None:
    # Descriptive binomial interval only: it does not account for source dependence.
    if not negative:
        return None
    from scipy.stats import beta
    return [float(beta.ppf(.025, fp, negative - fp + 1)) if fp else 0.,
            float(beta.ppf(.975, fp + 1, negative - fp)) if fp < negative else 1.]


def evaluate(labels: Sequence[str], groups: Sequence[str], scores: np.ndarray,
             thresholds: np.ndarray, targets: Sequence[str], *,
             bootstrap_draws: int = 2000, bootstrap_seed: int = 51106) -> dict:
    """Analyze already committed arrays, returning JSON-serializable results.

    Primary CIs are conditional on the fixed seeds, references, fitted matchers and
    calibration sets. No examples, seeds or reference episodes are selected here.
    Two 97.5% two-sided percentile CIs give nominal Bonferroni family coverage of
    95% for the two prespecified primary contrasts, subject to bootstrap validity.
    Secondary intervals are descriptive, unadjusted 95%. No thresholds are output.
    Use positive infinity for an unavailable/infinite frozen threshold, never NaN.
    The caller guarantees one row per unique query event and authentic group IDs.
    """
    started = time.monotonic()
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)
    targets = tuple(map(str, targets))
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if scores.ndim != 6 or scores.shape[:2] != (2, 2):
        raise ValueError("Expected scores[2,2,seed,reference,target,event]")
    if thresholds.shape != scores.shape[:-1] + (2,):
        raise ValueError("Threshold axes must match scores, with final alpha axis2")
    if len(set(targets)) != len(targets) or scores.shape[-2] != len(targets):
        raise ValueError("Targets must be unique and match the target axis")
    if labels.ndim != 1 or groups.shape != labels.shape or len(labels) != scores.shape[-1]:
        raise ValueError("Labels/groups must match the common event axis")
    if any(n == 0 for n in scores.shape) or not np.isfinite(scores).all():
        raise ValueError("Scores must be nonempty and finite")
    if np.isnan(thresholds).any() or np.isneginf(thresholds).any():
        raise ValueError("Thresholds may be finite or +infinity, not NaN/-infinity")
    if bootstrap_draws < 0:
        raise ValueError("bootstrap_draws must be nonnegative")
    group_names, group_index = np.unique(groups, return_inverse=True)
    group_count = len(group_names)
    if group_count < 2:
        raise ValueError("At least two independent sampling groups are required")
    shape = scores.shape[:-1]
    seeds, references = shape[2:4]
    one = np.ones(group_count, dtype=np.int64)
    cache = []
    point = np.empty(shape + (len(METRICS),), dtype=np.float64)
    per_configuration = []
    for index in np.ndindex(shape):
        training, matcher, seed, reference, target = index
        y = labels == targets[target]
        if not y.any() or y.all():
            raise ValueError(f"Target {targets[target]} lacks positives or negatives")
        ranking = _Ranking(y, scores[index], group_index, thresholds[index])
        cache.append((index, ranking))
        vector, rows = ranking.measure(one)
        point[index] = vector
        for row in rows:
            row["fpr_clopper_pearson_95_descriptive"] = _binomial_interval(row["fp"], row["negative"])
        per_configuration.append(dict(training=TRAINING[training], matcher=MATCHERS[matcher],
                                      seed_index=seed, reference_episode=reference,
                                      target=targets[target],
                                      metrics=dict(zip(METRICS, map(float, vector))),
                                      operating_points=rows))
    # Equally weight fixed targets, seeds and reference episodes, not their event counts.
    cell = point.mean(axis=(2, 3, 4))
    contrast = _contrasts(cell)
    draws = np.empty((bootstrap_draws, 2, 2, len(METRICS)), dtype=np.float64)
    valid = 0
    rng = np.random.default_rng(bootstrap_seed)
    # Counts by target/group allow undefined-draw checks before any ranking work.
    target_counts = np.stack([np.bincount(group_index, weights=(labels == t), minlength=group_count)
                              for t in targets])
    group_sizes = np.bincount(group_index, minlength=group_count)
    for _ in range(bootstrap_draws):
        counts = np.bincount(rng.integers(0, group_count, group_count), minlength=group_count)
        positives = target_counts @ counts
        total = group_sizes @ counts
        if np.any(positives == 0) or np.any(positives == total):
            continue  # Report skipped draws; never keep drawing until favorable.
        aggregate = np.zeros((2, 2, len(METRICS)))
        for index, ranking in cache:
            vector, _ = ranking.measure(counts)
            aggregate[index[:2]] += vector
        draws[valid] = aggregate / (seeds * references * len(targets))
        valid += 1
    draws = draws[:valid]
    contrast_draws = np.asarray([_contrasts(x) for x in draws]) if valid else np.empty((0, len(CONTRASTS), len(METRICS)))
    cell_rows = []
    for training in range(2):
        for matcher in range(2):
            cell_rows.append(dict(training=TRAINING[training], matcher=MATCHERS[matcher],
                                  metrics={name: dict(estimate=float(cell[training, matcher, k]),
                                                     paired_group_interval95=_interval(draws[:, training, matcher, k], .95))
                                           for k, name in enumerate(METRICS)}))
    secondary = []
    for j, name in enumerate(CONTRASTS):
        secondary.append(dict(contrast=name,
                              metrics={metric: dict(estimate=float(contrast[j, k]),
                                                    paired_group_interval95=_interval(contrast_draws[:, j, k], .95))
                                       for k, metric in enumerate(METRICS)}))
    k = METRICS.index("pauc_std_001")
    primary = []
    for j in (0, 1):
        interval = _interval(contrast_draws[:, j, k], .975)
        primary.append(dict(contrast=CONTRASTS[j], endpoint="standardized_pAUC_max_fpr_0.001",
                            estimate=float(contrast[j, k]), paired_group_interval975=interval,
                            lower_bound_above_zero=bool(interval and interval["low"] > 0)))
    # A simultaneous search has a different denominator from one-versus-rest macro FA.
    outside = ~np.isin(labels, targets)
    union_rows = []
    if outside.any():
        for tr, ma, se, re in np.ndindex(shape[:4]):
            for ai, alpha in enumerate(ALPHAS):
                decisions = scores[tr, ma, se, re] > thresholds[tr, ma, se, re, :, ai, None]
                alarms = np.any(decisions, axis=0)
                n = int(outside.sum()); fp = int(alarms[outside].sum())
                union_rows.append(dict(training=TRAINING[tr], matcher=MATCHERS[ma], seed_index=se,
                                       reference_episode=re, alpha=alpha, negative=n, fp=fp,
                                       fa_per10000=10000. * fp / n,
                                       fpr_clopper_pearson_95_descriptive=_binomial_interval(fp, n)))
    return dict(
        complete=True, seconds=time.monotonic() - started,
        axis_contract=dict(training=list(TRAINING), matcher=list(MATCHERS), seeds=seeds,
                           reference_episodes=references, targets=list(targets), alphas=list(ALPHAS)),
        cohort=dict(unique_query_events=len(labels), recording_groups=group_count,
                    target_counts={t: int((labels == t).sum()) for t in targets},
                    outside_target_events=int(outside.sum()),
                    repeated_evaluations_do_not_increase_event_count=True),
        primary=primary, cells=cell_rows, secondary_contrasts=secondary,
        per_configuration=per_configuration, any_target_background=union_rows,
        bootstrap=dict(requested=bootstrap_draws, valid=valid, skipped=bootstrap_draws - valid,
                       seed=bootstrap_seed, unit="paired detector-hour recording groups",
                       method="ordinary group resampling with replacement; shared draw across every cell/seed/reference/target",
                       conditional_on="fixed seeds, fixed reference episodes, FIT and CAL sets, fitted matchers and thresholds",
                       primary_interval="two 97.5% two-sided percentile intervals; nominal Bonferroni family 95% for two prespecified contrasts",
                       secondary_interval="unadjusted 95% percentile intervals; descriptive"),
        definitions=dict(
            training_main="mean(trained-random), equally averaging both matchers, fixed seeds, reference episodes and targets",
            matcher_main="mean(negative_aware-cosine), equally averaging both training conditions, fixed seeds, references and targets",
            interaction="(trained_negative_aware-random_negative_aware)-(trained_cosine-random_cosine)",
            pAUC="McClish standardized partial ROC area; tied-score groups; random diagonal=.5, perfect=1",
            empirical_recall="maximum observed recall with empirical FPR at most budget; descriptive ranking, no deployable threshold output",
            decisions="score strictly greater than frozen CAL threshold; no test recalibration",
            cells="equal mean of per-target/per-seed/per-reference metrics, not pooled duplicate observations"),
        limitations=[
            "Group intervals condition on the prespecified seeds and reference choices; they do not establish performance across all future seeds/references.",
            "Nominal bootstrap coverage is approximate, particularly for sparse tail false positives; nominal alpha is not an achieved-rate guarantee.",
            "Binomial FPR bounds are descriptive and ignore residual temporal/detector dependence; use recording-group intervals for paired comparisons.",
            "Fewer than 1000 negatives makes one false positive exceed the .001 empirical budget; partial ROC area interpolates between attainable points.",
            "Label-informed cohort sampling must be disclosed; prevalence-dependent metrics need not represent an archive population.",
            "The negative-aware-versus-cosine contrast includes use of FIT negatives; cosine does not fit them.",
            "A main effect does not show that both conditional simple effects improve; report the full factorial results.",
            "No findings here establish component novelty, new scientific discoveries or competition outcomes.",
        ],
    )


def sanity_checks() -> dict:
    """Synthetic arrays only; independent sklearn ROC and weighted-count checks."""
    from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
    rng = np.random.default_rng(26)
    y = np.arange(120) % 3 == 0
    groups = np.arange(len(y))
    checked = 0
    for score in (np.zeros(120), y.astype(float), -y.astype(float),
                  rng.normal(size=120), rng.integers(0, 8, size=120).astype(float)):
        for weights in (np.ones(120, dtype=int), rng.integers(0, 5, size=120)):
            cuts = np.array([.5, np.inf])
            got, rows = _Ranking(y, score, groups, cuts).measure(weights)
            for k, maximum in ((0, None), (1, .001), (2, .01)):
                expected = roc_auc_score(y, score, sample_weight=weights, max_fpr=maximum)
                assert np.isclose(got[k], expected, atol=1e-12), (got[k], expected)
                checked += 1
            for cut, row in zip(cuts, rows):
                tn, fp, fn, tp = confusion_matrix(y, score > cut, sample_weight=weights, labels=[False, True]).ravel()
                assert (row['tp'], row['fp'], row['tn'], row['fn']) == (tp, fp, tn, fn)
                assert np.isclose(row['mcc'], matthews_corrcoef(y, score > cut, sample_weight=weights))
    z = np.zeros((2, 2, 1)); z[1] = 2; z[:, 1] += 3
    assert np.array_equal(_contrasts(z).ravel(), [2, 3, 0, 2, 2, 3, 3])
    return dict(passed=True, weighted_ROC_comparisons=checked,
                cases=["all ties", "perfect", "reversed", "continuous", "mixed ties",
                       "zero and repeated bootstrap weights", "strict threshold ties", "infinite threshold",
                       "independent sklearn confusion/MCC", "factorial contrast signs"])
