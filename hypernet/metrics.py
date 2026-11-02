"""Binary same-instance metrics with development-only threshold selection.

Scores are arbitrary finite ranking scores, not assumed probabilities. Acceptance
always means score >= threshold. Undefined metrics are represented by None.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterable
import numpy as np


def _arrays(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equally sized one-dimensional arrays")
    if not np.all(np.isin(labels, [False, True, 0, 1])):
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all():
        raise ValueError("All scores must be finite; record failed predictions separately")
    return labels.astype(bool), scores


def binary_auroc(labels, scores):
    """Mann-Whitney AUROC, assigning half credit to tied positive/negative pairs."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(s, kind="stable")
    ys, ss = y[order], s[order]
    rank_sum = 0.0
    start = 0
    while start < len(ss):
        end = start + 1
        while end < len(ss) and ss[end] == ss[start]:
            end += 1
        rank_sum += float(ys[start:end].sum()) * (start + 1 + end) / 2
        start = end
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def binary_average_precision(labels, scores):
    """Non-interpolated AP: recall increments times precision, with ties grouped."""
    y, s = _arrays(labels, scores)
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    true_pos = 0
    ap = 0.0
    start = 0
    while start < len(ss):
        end = start + 1
        while end < len(ss) and ss[end] == ss[start]:
            end += 1
        added = int(ys[start:end].sum())
        true_pos += added
        ap += (added / positives) * (true_pos / end)
        start = end
    return float(ap)


def binary_metrics(labels, scores, threshold):
    y, s = _arrays(labels, scores)
    if not isinstance(threshold, (int, float, np.number)) or math.isnan(float(threshold)):
        raise ValueError("Threshold must be a numeric value, not NaN")
    pred = s >= float(threshold)
    tp = int(np.sum(pred & y)); fn = int(np.sum(~pred & y))
    fp = int(np.sum(pred & ~y)); tn = int(np.sum(~pred & ~y))
    tpr = tp / (tp + fn) if tp + fn else None
    fpr = fp / (fp + tn) if fp + tn else None
    precision = tp / (tp + fp) if tp + fp else None
    balanced = (tpr + 1 - fpr) / 2 if tpr is not None and fpr is not None else None
    return {
        "n": len(y), "n_positive": tp + fn, "n_negative": fp + tn,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": tpr, "fpr": fpr, "precision": precision,
        "accuracy": float(np.mean(pred == y)) if len(y) else None,
        "balanced_accuracy": balanced,
        "auroc": binary_auroc(y, s),
        "average_precision": binary_average_precision(y, s),
    }


def calibrate_threshold(dev_rows: Iterable[dict], target_fpr: float = 0.05) -> dict:
    """Smallest score cutoff achieving the empirical development FPR constraint.

    Strictly disallows test records. The result is not a population FPR guarantee.
    Ties are kept together. With few negatives, the allowed error count can be 0.
    """
    rows = list(dev_rows)
    if not rows:
        raise ValueError("Threshold calibration requires nonempty development predictions")
    if any(r.get("split") != "dev" for r in rows):
        raise ValueError("Threshold calibration accepts split='dev' records only")
    if not math.isfinite(target_fpr) or not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be finite and in [0, 1)")
    y, s = _arrays([r["present"] for r in rows], [r["score"] for r in rows])
    negatives = s[~y]
    if not len(negatives):
        raise ValueError("Development target-absent examples are required for FPR calibration")
    allowed = int(math.floor(target_fpr * len(negatives) + 1e-12))
    # The boundary is the (allowed+1)-th highest negative. Reject that entire
    # score group, including ties, rather than accepting more errors than allowed.
    boundary = float(np.sort(negatives)[::-1][allowed])
    threshold = float(np.nextafter(boundary, math.inf))
    if not math.isfinite(threshold):
        raise ValueError("Scores too large to form a finite rejection threshold")
    metrics = binary_metrics(y, s, threshold)
    return {
        "threshold": threshold, "rule": "score >= threshold",
        "target_fpr": float(target_fpr), "n_dev": len(rows),
        "n_dev_negative": len(negatives), "allowed_false_positives": allowed,
        "empirical_dev_fpr": metrics["fpr"], "empirical_dev_tpr": metrics["tpr"],
        "calibration_object_ids": sorted({str(r["object_id"]) for r in rows}),
        "interpretation": "Empirical development constraint only; test FPR can be higher.",
    }


def object_cluster_bootstrap(rows, threshold, samples=500, seed=17, confidence=0.95):
    """Resample target objects, retaining all positive and absent pairs per object.

    This targets uncertainty over sampled reference identities. It does not account
    for every dependence caused by reusing distractors across reference identities.
    Threshold is held fixed; threshold-selection uncertainty is not bootstrapped.
    """
    if samples < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    groups = defaultdict(list)
    for r in rows:
        if "object_id" not in r:
            raise ValueError("object_id is required for clustered uncertainty")
        groups[str(r["object_id"])].append(r)
    names = sorted(groups)
    keys = ["tpr", "fpr", "balanced_accuracy", "auroc", "average_precision"]
    result = {"method": "target-object percentile bootstrap", "n_objects": len(names),
              "samples_requested": int(samples), "confidence": confidence,
              "seed": int(seed), "threshold_fixed": True, "intervals": {}}
    if len(names) < 2 or samples == 0:
        result["reason"] = "At least two objects and a positive replicate count are required"
        result["intervals"] = {key: None for key in keys}
        return result
    rng = np.random.default_rng(seed)
    values = {key: [] for key in keys}
    for _ in range(samples):
        drawn = rng.choice(names, len(names), replace=True)
        sample_rows = [r for name in drawn for r in groups[name]]
        metric = binary_metrics([r["present"] for r in sample_rows],
                                [r["score"] for r in sample_rows], threshold)
        for key in keys:
            if metric[key] is not None:
                values[key].append(metric[key])
    alpha = (1 - confidence) / 2
    for key, vals in values.items():
        result["intervals"][key] = ({"low": float(np.quantile(vals, alpha)),
                                     "high": float(np.quantile(vals, 1-alpha)),
                                     "valid_replicates": len(vals)} if vals else None)
    return result


def evaluate_scores(rows, threshold, bootstrap_samples=500, seed=17):
    """Evaluate provided records at a fixed cutoff; never selects a threshold."""
    rows = list(rows)
    if not rows:
        raise ValueError("Evaluation requires at least one prediction")
    overall = binary_metrics([r["present"] for r in rows], [r["score"] for r in rows], threshold)
    subsets = defaultdict(list)
    for row in rows:
        subsets[str(row.get("condition", "unspecified"))].append(row)
    by_condition = {
        condition: binary_metrics([r["present"] for r in subset], [r["score"] for r in subset], threshold)
        for condition, subset in sorted(subsets.items())
    }
    return {"threshold": float(threshold), "accept_rule": "score >= threshold",
            "splits": sorted({str(r.get("split", "unspecified")) for r in rows}),
            "overall": overall, "by_condition": by_condition,
            "object_cluster_ci": object_cluster_bootstrap(rows, threshold, bootstrap_samples, seed)}


def box_iou(box_a, box_b):
    """Continuous-coordinate xyxy IoU; no +1 pixel convention."""
    a, b = np.asarray(box_a, float), np.asarray(box_b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("Boxes must be finite [x0,y0,x1,y1] arrays")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("Box corners are reversed")
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b=(b[2]-b[0])*(b[3]-b[1])
    union = area_a+area_b-inter
    return float(inter / union) if union > 0 else 0.
