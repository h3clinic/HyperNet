"""Independently audit completed, committed factorial results; never runs models.

Explicit --run is required. The complete results gate is checked before opening
any query scores. This recomputes point metrics with sklearn, without bootstrap,
threshold changes, data selection, training or neural inference.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
EXPECTED_PROTOCOL = "52acaeca63bf959e340fecaac65dd796a6de5920409a1c217172e160cff33380"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def write_new(name, value):
    with (R / name).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run():
    # Do not inspect partially scored query arrays.
    results = read("results.json")
    if results.get("complete") is not True:
        raise RuntimeError("Wait for all final results before independent analysis")
    for name in ("independent_result_audit.json", "detector_descriptive.json"):
        if (R / name).exists():
            raise FileExistsError("Preserve completed independent analysis: " + name)
    protocol, preflight = read("protocol.json"), read("independent_preflight.json")
    assert sha(R / "protocol.json") == EXPECTED_PROTOCOL
    assert results["protocol_sha256"] == preflight["protocol_sha256"] == EXPECTED_PROTOCOL
    assert preflight["passed"] is True
    supplement = preflight["descriptive_post_protocol_supplement"]
    assert sha(supplement["path"]) == supplement["sha256"]
    for name, expected in preflight["review_hashes"].items():
        assert sha(R / name) == expected
    for name in ("labels.json", "factorial_metrics.py", "factorial_core.py", "run_experiment.py"):
        assert sha(R / name) == protocol["source_hashes"][str(R / name)]
    commitment, fit = read("score_commitment.json"), read("fit_commitment.json")
    assert commitment["protocol_sha256"] == fit["protocol_sha256"] == EXPECTED_PROTOCOL
    assert commitment["all_conditions_scored_before_metrics"] is True
    assert results["score_commitment_sha256"] == sha(R / "score_commitment.json")
    for file, expected in commitment["files"].items():
        assert sha(file) == expected, file
    assert sha(R / "thresholds.npy") == fit["files"][str(R / "thresholds.npy")]
    acquisition = read("acquisition.json")
    assert acquisition["complete"] is True and acquisition["protocol_sha256"] == EXPECTED_PROTOCOL
    times = {key: datetime.fromisoformat(value) for key, value in {
        "preflight": preflight["completed_utc"],
        "fit": fit["committed_utc"],
        "acquisition": acquisition["completed_utc"],
        "scores": commitment["committed_utc"],
        "results": results["completed_utc"],
    }.items()}
    assert times["preflight"] < times["scores"] <= times["results"]
    assert times["fit"] <= times["scores"] and times["acquisition"] <= times["scores"]
    records, label_rows = read("records_unlabelled.json"), read("labels.json")
    lookup = {row["event_id"]: row["label"] for row in label_rows}
    assert len(lookup) == len(label_rows)
    assert len({row["event_id"] for row in records}) == len(records)
    labels = np.asarray([lookup[row["event_id"]] for row in records])
    detectors = np.asarray([row["ifo"] for row in records])
    groups = np.asarray([row["ifo"] + "_" + str(int(float(row["event_time"]) // 3600)) for row in records])
    scores, thresholds = np.load(R / "scores.npy"), np.load(R / "thresholds.npy")
    assert list(scores.shape) == commitment["shape"]
    assert scores.shape[-1] == len(records) == results["cohort"]["unique_query_events"]
    axis = results["axis_contract"]
    targets, training, matchers, alphas = (axis[key] for key in ("targets", "training", "matcher", "alphas"))
    assert training == ["random", "trained"] and matchers == ["cosine", "negative_aware"]
    assert scores.shape[:-1] == (2, 2, 3, 5, 4) and alphas == [.001, .01]
    assert thresholds.shape == scores.shape[:-1] + (2,)

    from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef
    rows = {(row["training"], row["matcher"], row["seed_index"], row["reference_episode"], row["target"]): row
            for row in results["per_configuration"]}
    assert len(rows) == int(np.prod(scores.shape[:-1]))
    p001 = np.empty(scores.shape[:-1], dtype=np.float64)
    maximum_error, confusion_checks, ranking_checks = 0., 0, 0
    for tr, ma, se, re, ti in np.ndindex(scores.shape[:-1]):
        index = (tr, ma, se, re, ti)
        y, score = labels == targets[ti], scores[index]
        row = rows[(training[tr], matchers[ma], se, re, targets[ti])]
        for metric, limit in (("auroc", None), ("pauc_std_001", .001), ("pauc_std_01", .01)):
            value = float(roc_auc_score(y, score, max_fpr=limit))
            error = abs(value - row["metrics"][metric])
            assert error <= 1e-12, (index, metric, value, row["metrics"][metric])
            maximum_error = max(maximum_error, error)
            ranking_checks += 1
            if metric == "pauc_std_001":
                p001[index] = value
        for ai, operating in enumerate(row["operating_points"]):
            assert operating["alpha"] == alphas[ai]
            predicted = score > thresholds[index + (ai,)]
            tn, fp, fn, tp = map(int, confusion_matrix(y, predicted, labels=[False, True]).ravel())
            assert (tp, fp, tn, fn) == tuple(operating[key] for key in ("tp", "fp", "tn", "fn"))
            p, n = tp + fn, tn + fp
            expected = dict(balanced_accuracy=.5 * (tp / p + tn / n),
                            mcc=float(matthews_corrcoef(y, predicted)), recall=tp / p,
                            specificity=tn / n, fa_per10000=10000 * fp / n)
            for metric, value in expected.items():
                assert abs(value - operating[metric]) <= 1e-12, (index, ai, metric)
            confusion_checks += 1
    cells = p001.mean(axis=(2, 3, 4))
    cell_rows = []
    for tr, ma in np.ndindex((2, 2)):
        reported = next(row for row in results["cells"]
                        if row["training"] == training[tr] and row["matcher"] == matchers[ma])
        assert abs(cells[tr, ma] - reported["metrics"]["pauc_std_001"]["estimate"]) <= 1e-12
        cell_rows.append(dict(training=training[tr], matcher=matchers[ma],
                              sklearn_macro_pauc_std_001=float(cells[tr, ma])))
    primary = dict(training_main=float(np.mean(cells[1] - cells[0])),
                   matcher_main=float(np.mean(cells[:, 1] - cells[:, 0])))
    for row in results["primary"]:
        assert abs(row["estimate"] - primary[row["contrast"]]) <= 1e-12

    spec = importlib.util.spec_from_file_location("verified_detector_supplement", supplement["path"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    descriptive = helper.summarize_by_detector(labels, detectors, groups, scores, thresholds, targets)
    bindings = {name: sha(R / name) for name in (
        "protocol.json", "independent_preflight.json", "results.json", "score_commitment.json",
        "fit_commitment.json", "scores.npy", "thresholds.npy", "records_unlabelled.json", "labels.json",
        "factorial_metrics.py", "detector_summary.py", "independent_result_audit.py")}
    descriptive["provenance"] = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                    source_hashes=bindings, new_inference=False, bootstrap_rerun=False,
                                    code_predeclared_in="independent_preflight.json descriptive supplement")
    write_new("detector_descriptive.json", descriptive)
    audit = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
                 source_hashes=bindings, detector_descriptive_sha256=sha(R / "detector_descriptive.json"),
                 gate="Completed results checked before any query-score access",
                 new_inference=False, new_training=False, bootstrap_rerun=False,
                 threshold_changes=False, sample_selection=False,
                 checks=dict(protocol_and_score_commitment_hashes=True, frozen_threshold_hash=True,
                             helper_hash_bound_in_preflight=True, chronology=True,
                             labels_joined_by_unique_event_id=True,
                             sklearn_ranking_metrics=ranking_checks,
                             sklearn_fixed_confusion_checks=confusion_checks,
                             maximum_ranking_metric_absolute_error=maximum_error,
                             macro_primary_cell_and_contrast_estimates=True),
                 cells=cell_rows, primary_point_estimates=primary,
                 limitations=["No independent bootstrap rerun; uncertainty is from the frozen original analysis.",
                              "Detector-stratified comparisons are descriptive and use unchanged calibration thresholds."])
    write_new("independent_result_audit.json", audit)
    print(json.dumps(dict(passed=True, cells=cell_rows, primary=primary,
                          ranking_checks=ranking_checks, confusion_checks=confusion_checks,
                          detector_summary=str(R / "detector_descriptive.json")), indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        run()
    else:
        parser.print_help()
