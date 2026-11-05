"""Descriptive detector-stratified summaries of already committed factorial scores.

No inference, fitting, threshold selection, files or outcome-based selection.
Every stratum reuses all fixed seeds/references and the original CAL thresholds.
The caller must preserve the committed common event order when joining labels,
detector identifiers and recording groups. This is a descriptive supplement;
it neither adds primary tests nor changes the pooled factorial analysis.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np


def _metrics():
    path = Path(__file__).with_name("factorial_metrics.py")
    spec = importlib.util.spec_from_file_location("factorial_detector_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_by_detector(labels: Sequence[str], detectors: Sequence[str],
                          groups: Sequence[str], scores: np.ndarray,
                          thresholds: np.ndarray, targets: Sequence[str]) -> dict:
    """Return point estimates only, using the audited common metric implementation.

    Arrays retain [random/trained, cosine/negative_aware, seed, reference, target,
    event] axes. No reference/seed is selected per detector. A stratum missing
    any target's positives or negatives has an unavailable full-target macro;
    its denominator is never silently changed.
    """
    labels = np.asarray(labels, dtype=str)
    detectors = np.asarray(detectors, dtype=str)
    groups = np.asarray(groups, dtype=str)
    scores = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    targets = tuple(map(str, targets))
    if (labels.ndim != 1 or detectors.shape != labels.shape or groups.shape != labels.shape
            or scores.ndim != 6 or scores.shape[:2] != (2, 2)
            or scores.shape[-2:] != (len(targets), len(labels))
            or thresholds.shape != scores.shape[:-1] + (2,)):
        raise ValueError("Detector summary inputs do not match the factorial event contract")
    if set(detectors) - {"H1", "L1"}:
        raise ValueError("Expected official H1/L1 detector identifiers")
    if not np.isfinite(scores).all() or np.isnan(thresholds).any() or np.isneginf(thresholds).any():
        raise ValueError("Scores must be finite; thresholds finite or positive infinity")
    metrics = _metrics()
    strata = []
    for detector in ("H1", "L1"):
        selected = detectors == detector
        stratum_labels = labels[selected]
        counts = {target: int((stratum_labels == target).sum()) for target in targets}
        invalid = [target for target, count in counts.items()
                   if count == 0 or count == len(stratum_labels)]
        summary = dict(detector=detector, unique_query_events=int(selected.sum()),
                       recording_groups=len(np.unique(groups[selected])), target_counts=counts,
                       outside_target_events=int((~np.isin(stratum_labels, targets)).sum()),
                       same_reference_episodes_and_seeds_as_pooled=True,
                       thresholds_reused_unchanged=True, detector_recalibration=False)
        if invalid or summary["recording_groups"] < 2:
            summary.update(available=False,
                           reason="At least one target lacks positives/negatives, or fewer than two groups",
                           unavailable_targets=invalid, all_target_macro=None)
        else:
            result = metrics.evaluate(stratum_labels, groups[selected], scores[..., selected],
                                      thresholds, targets, bootstrap_draws=0)
            cells = [dict(training=row["training"], matcher=row["matcher"],
                          metrics={key: value["estimate"] for key, value in row["metrics"].items()})
                     for row in result["cells"]]
            summary.update(available=True, cells=cells,
                           per_configuration=result["per_configuration"],
                           any_target_background=result["any_target_background"])
        strata.append(summary)
    return dict(descriptive_only=True, no_new_primary_tests=True, no_tuning=True,
                unique_query_events=len(labels), targets=list(targets),
                seed_count=scores.shape[2], reference_episode_count=scores.shape[3],
                detector_strata=strata,
                limitations=[
                    "Differences between detectors are descriptive; no detector-specific model, reference or threshold is selected.",
                    "Sparse positives make stratum recall unstable, particularly the ten planned Livingston Low Frequency Burst positives.",
                    "Prevalence and negative composition differ by detector; MCC and false-alarm rates are conditional on this sampled cohort.",
                    "The original pooled paired group analysis remains primary. These point summaries do not establish an interaction with detector.",
                    "Per-configuration binomial false-alarm bounds inherited from the metric module are descriptive and ignore residual dependence.",
                ])


def self_test() -> dict:
    """Synthetic arrays only: exact strata, frozen thresholds and missing target."""
    targets = ["a", "b", "c", "d"]
    labels = np.asarray((targets + ["other"]) * 8)
    detectors = np.asarray(["H1"] * 20 + ["L1"] * 20)
    groups = np.asarray([str(i) for i in range(40)])
    base = np.stack([labels == target for target in targets]).astype(float)
    scores = np.broadcast_to(base, (2, 2, 1, 2, 4, 40)).copy()
    thresholds = np.broadcast_to([.5, np.inf], (2, 2, 1, 2, 4, 2)).copy()
    result = summarize_by_detector(labels, detectors, groups, scores, thresholds, targets)
    for stratum in result["detector_strata"]:
        assert stratum["available"] and stratum["unique_query_events"] == 20
        assert stratum["outside_target_events"] == 4
        for cell in stratum["cells"]:
            assert cell["metrics"]["balanced_accuracy_alpha001"] == 1
            assert cell["metrics"]["fa_per10000_alpha001"] == 0
            assert cell["metrics"]["recall_alpha01"] == 0
    missing = labels.copy()
    missing[(detectors == "L1") & (labels == "a")] = "other"
    result_missing = summarize_by_detector(missing, detectors, groups, scores, thresholds, targets)
    assert result_missing["detector_strata"][0]["available"]
    assert not result_missing["detector_strata"][1]["available"]
    assert result_missing["detector_strata"][1]["unavailable_targets"] == ["a"]
    return dict(passed=True, synthetic_arrays_only=True, no_neural_inference=True,
                exact_detector_counts=True, frozen_threshold_behavior=True,
                missing_target_does_not_change_macro_denominator=True)


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    else:
        parser.print_help()
