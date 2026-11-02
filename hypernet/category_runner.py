"""One-shot category-presence evaluation with development-category calibration.

The model sees only one reference RGB array and one query RGB array at prediction
time. Category labels and optional object annotations remain evaluation metadata.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import time

import numpy as np

from .data import load_rgb
from .metrics import binary_metrics, calibrate_threshold
from .runner import json_safe, write_json

METRIC_KEYS = ("tpr", "fpr", "balanced_accuracy", "auroc", "average_precision")


def _metrics(rows, threshold):
    return binary_metrics([r["present"] for r in rows], [r["score"] for r in rows], threshold)


def _macro(category_metrics):
    """Equal weight for each category with a defined value; report denominators."""
    result = {}
    counts = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in category_metrics if m[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
    result["defined_categories"] = counts
    result["n_categories"] = len(category_metrics)
    return result


def category_cluster_bootstrap(rows, threshold, samples=500, seed=17, confidence=.95):
    """Bootstrap category-level metrics, retaining every episode within a category.

    The intervals describe variation over target categories. Shared distractor
    categories/photos create additional dependence not fully captured here.
    Threshold selection and external pretraining uncertainty are not resampled.
    """
    if not isinstance(samples, (int, np.integer)) or samples < 0 or not 0 < confidence < 1:
        raise ValueError("Invalid category-bootstrap settings")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["category_id"])].append(row)
    names = sorted(groups)
    result = {
        "method": "target-category percentile bootstrap of category-macro metrics",
        "n_categories": len(names), "samples_requested": int(samples),
        "confidence": float(confidence), "seed": int(seed),
        "threshold_fixed": True, "all_support_episodes_retained": True,
        "interpretation": "Conditional on fixed development cutoff; shared distractor categories/images may induce cross-cluster dependence.",
        "intervals": {},
    }
    if len(names) < 2 or samples == 0:
        result["reason"] = "At least two categories and a positive replicate count are required"
        result["intervals"] = {key: None for key in METRIC_KEYS}
        return result
    metrics = [_metrics(groups[name], threshold) for name in names]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(samples, len(names)))
    alpha = (1-confidence)/2
    for key in METRIC_KEYS:
        values = np.array([np.nan if m[key] is None else m[key] for m in metrics])
        sampled = values[indices]
        counts = np.sum(np.isfinite(sampled), axis=1)
        valid = counts > 0
        means = np.nansum(sampled[valid], axis=1)/counts[valid]
        result["intervals"][key] = (
            {"low": float(np.quantile(means, alpha)),
             "high": float(np.quantile(means, 1-alpha)),
             "valid_replicates": int(valid.sum())} if valid.any() else None)
    return result


def evaluate_category_scores(rows, threshold, bootstrap_samples=500, seed=17):
    """Evaluate a fixed cutoff; no calibration or model selection occurs here."""
    rows = list(rows)
    if not rows:
        raise ValueError("Category evaluation requires prediction rows")
    if not math.isfinite(float(threshold)):
        raise ValueError("Category evaluation requires a finite threshold")
    by_category, by_episode, by_condition = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        if not row.get("category_id") or not row.get("episode_id"):
            raise ValueError("category_id and episode_id are required")
        by_category[str(row["category_id"])].append(row)
        by_episode[str(row["episode_id"])].append(row)
        by_condition[str(row.get("condition", "unspecified"))].append(row)
    categories = {key: _metrics(value, threshold) for key, value in sorted(by_category.items())}
    episodes = {}
    for key, value in sorted(by_episode.items()):
        cats = {r["category_id"] for r in value}
        supports = {r["reference"] for r in value}
        if len(cats) != 1 or len(supports) != 1:
            raise ValueError("Each episode must have one target category and one support photograph")
        episodes[key] = {
            "category_id": str(next(iter(cats))),
            "reference": next(iter(supports)),
            "support_id": value[0].get("support_id"),
            "metrics": _metrics(value, threshold),
        }
    sensitivity = {}
    for category in sorted(categories):
        selected_ids = [key for key, ep in episodes.items() if ep["category_id"] == category]
        selected = [episodes[key]["metrics"] for key in selected_ids]
        query_sets = []
        for key in selected_ids:
            episode_rows = by_episode[key]
            identifiers = [r.get("query_image_id", r.get("query")) for r in episode_rows]
            query_sets.append(None if any(i is None for i in identifiers) else
                              frozenset((i, r["present"]) for i, r in zip(identifiers, episode_rows)))
        comparable = (len(query_sets) > 1 and all(q is not None for q in query_sets) and
                      all(q == query_sets[0] for q in query_sets[1:]))
        spread = {"n_support_episodes": len(selected),
                  "identical_query_sets": comparable,
                  "interpretation": ("Same query photographs and labels: observed variation is across support photos and their reference-dependent adaptation at fixed settings."
                                     if comparable else
                                     "Query sets are different, unspecified, or only one support episode exists; do not interpret this spread as isolated support-photo sensitivity.")}
        for key in ("tpr", "fpr", "balanced_accuracy"):
            values = [m[key] for m in selected if m[key] is not None]
            spread[key] = (
                {"mean": float(np.mean(values)), "min": float(min(values)),
                 "max": float(max(values)), "std": float(np.std(values)),
                 "defined_episodes": len(values)} if values else None)
        sensitivity[category] = spread
    return {
        "threshold": float(threshold), "accept_rule": "score >= threshold",
        "splits": sorted({str(r.get("split", "unspecified")) for r in rows}),
        "overall": _metrics(rows, threshold),
        "category_macro": _macro(list(categories.values())),
        "by_category": categories, "by_episode": episodes,
        "by_condition": {key: _metrics(value, threshold) for key, value in sorted(by_condition.items())},
        "support_sensitivity": {
            "per_category": sensitivity,
            "interpretation": "Observed spread across predefined one-support episodes. Per-category identical_query_sets states whether queries were held fixed. No best-reference selection is applied.",
        },
        "category_cluster_ci": category_cluster_bootstrap(rows, threshold, bootstrap_samples, seed),
    }


def _validate_evaluation_splits(pairs):
    splits = {r["split"] for r in pairs}
    if not {"dev", "test"}.issubset(splits):
        raise ValueError("Both development and test category partitions are required")
    assignments = defaultdict(set)
    for row in pairs:
        assignments[str(row["category_id"])].add(row["split"])
    if any(len(s) != 1 for s in assignments.values()):
        raise ValueError("Categories must be disjoint across evaluation splits")
    for row in pairs:
        query_categories = row.get("query_category_ids")
        if query_categories is None:
            query_categories = [row["query_category_id"]]
        if (not isinstance(query_categories, list) or
                any(not isinstance(c, str) or not c for c in query_categories)):
            raise ValueError("query_category_ids must be a list of nonempty category strings")
        if row["present"] != (row["category_id"] in query_categories):
            raise ValueError("Category-presence label disagrees with query category membership")
        for category in query_categories:
            # Non-target background categories may be shared. A selected final
            # target category must not appear in development query annotations.
            if category in assignments and row["split"] not in assignments[category]:
                raise ValueError("Selected categories must be disjoint across evaluation splits, including query contents")
    return {split: sorted(k for k, v in assignments.items() if split in v) for split in sorted(splits)}


def run_category_evaluation(manifest_path, models, output_dir, target_fpr=.05,
                            bootstrap_samples=500, seed=17):
    """Run full-image one-shot category presence with one shared cutoff per method.

    All references (including test supports) may be enrolled. Only development
    category scores calibrate the global cutoff. Query images/labels are never
    supplied to prepare(), and labels/boxes are never supplied to predict().
    """
    from .category_data import load_category_manifest

    if not models:
        raise ValueError("At least one category model is required")
    if not math.isfinite(target_fpr) or not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be finite and in [0,1)")
    if not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_category_manifest(manifest_path)
    root = Path(manifest["root"])
    pairs = manifest["pairs"]
    category_splits = _validate_evaluation_splits(pairs)
    # Train rows are retained as a distinct resource in some schemas, but this
    # evaluator intentionally accepts only its predefined dev/test episodes.
    if any(r["split"] not in {"dev", "test"} for r in pairs):
        raise ValueError("Category evaluation accepts dev/test pairs only")
    for name, model in models.items():
        if getattr(model, "calibration_mode", None) is not None:
            raise ValueError(f"{name}: category evaluator requires a single comparable raw score, not route-specific calibration")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty category-results directory to preserve earlier evidence")
    output.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ("numpy", "Pillow", "opencv-python-headless", "torch", "torchvision", "hypernet-starter"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {
        "completed": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "image-conditioned one-shot category presence",
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "schema_version": manifest.get("schema_version"),
        "protocol": manifest.get("protocol"),
        "dataset_notes": manifest.get("notes", []),
        "dataset_limitations": manifest.get("limitations", []),
        "annotation_budget": manifest.get("annotation_budget", manifest.get("supervision", {})),
        "support_preprocessing": manifest.get("support_preprocessing", "Reference RGB is loaded exactly as supplied by the dataset manifest"),
        "validation": manifest["validation"],
        "category_splits": category_splits,
        "target_dev_fpr": float(target_fpr), "seed": int(seed),
        "versions": versions, "platform": platform.platform(),
        "budget": {
            "support_photos_per_episode": 1,
            "distinct_support_photos": len({r["reference"] for r in pairs}),
            "episodes": len({r["episode_id"] for r in pairs}),
            "development_target_categories": len({r["category_id"] for r in pairs if r["split"] == "dev"}),
            "test_target_categories": len({r["category_id"] for r in pairs if r["split"] == "test"}),
            "annotation_input_to_model": "Reference RGB as supplied by dataset, full query RGB only. Support cropping/annotation resources must be counted from manifest provenance; query labels and boxes are never model inputs.",
            "reference_repetition": "Independent one-support episodes; their support photos are not jointly supplied to a model prediction",
        },
        "models": {},
        "interpretation": [
            "Category labels define presence; same-category independently sourced photos are positives.",
            "Physical-instance uniqueness is not established when instance IDs are unknown.",
            "One target support photo does not mean one image of total visual training; inspect model provenance.",
            "The global cutoff is fitted only on development categories. Its empirical FPR target is not a test guarantee.",
            "The full-query-image track supplies no ground-truth query crops, masks, category labels or text prompts to models. Support preprocessing may use an explicitly counted reference annotation.",
            "Category-level uncertainty is conditional on the fixed cutoff and may miss dependence from shared distractors.",
            "Pair timings include cache effects and exclude file loading, model construction and explicit support enrollment.",
        ],
    }
    write_json(output/"summary.json", summary)
    images = {}
    def image(relative):
        if relative not in images:
            images[relative] = load_rgb(root/relative)
        return images[relative]

    all_rows = []
    with (output/"predictions.partial.jsonl").open("w", encoding="utf-8") as stream:
        for alias, model in models.items():
            print(f"CATEGORY MODEL {alias}: {len(pairs)} pairs", flush=True)
            adaptation_records = []
            if hasattr(model, "prepare"):
                for reference in sorted({r["reference"] for r in pairs}):
                    before = time.perf_counter()
                    model.prepare(image(reference))
                    adaptation_records.append({"reference": reference, "seconds": time.perf_counter()-before})
            rows = []
            for index, pair in enumerate(pairs):
                reference, query = image(pair["reference"]), image(pair["query"])
                before = time.perf_counter()
                prediction = model.predict(reference, query)
                elapsed = time.perf_counter()-before
                if not math.isfinite(float(prediction.score)):
                    raise ValueError(f"{alias}: nonfinite prediction on {pair['id']}")
                row = json_safe({**pair, "model": alias, "score": float(prediction.score),
                                 "latency_ms": 1000*elapsed, "prediction": prediction.to_dict()})
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                rows.append(row)
                if (index+1) % 100 == 0:
                    print(f"  {index+1}/{len(pairs)}", flush=True)
            # Legacy calibrator's object_id label is an internal adapter only.
            # Its output is renamed so the category task never claims object IDs.
            dev = [{**r, "object_id": r["category_id"]} for r in rows if r["split"] == "dev"]
            calibration = calibrate_threshold(dev, target_fpr)
            calibration["calibration_category_ids"] = calibration.pop("calibration_object_ids")
            calibration["scope"] = "Single global cutoff across development target categories and predefined support episodes"
            cutoff = calibration["threshold"]
            for row in rows:
                row["accepted"] = row["score"] >= cutoff
            test = [r for r in rows if r["split"] == "test"]
            metrics = evaluate_category_scores(test, cutoff, bootstrap_samples, seed)
            dev_metrics = evaluate_category_scores([r for r in rows if r["split"] == "dev"], cutoff, 0, seed)
            latency = [r["latency_ms"] for r in test]
            summary["models"][alias] = {
                "provenance": model.provenance(), "calibration": calibration,
                "development_diagnostics": dev_metrics, "test": metrics,
                "adaptation_seconds": sum(r["seconds"] for r in adaptation_records),
                "adaptation_records": adaptation_records,
                "latency_ms": {"median": float(np.median(latency)), "p95": float(np.quantile(latency, .95))},
            }
            all_rows.extend(rows)
            write_json(output/"summary.json", summary)
            print(f"  test macro TPR={metrics['category_macro']['tpr']}, FPR={metrics['category_macro']['fpr']}", flush=True)
    fields = ["model", "id", "category_id", "query_category_id", "query_category_ids", "episode_id", "support_id",
              "reference", "query", "split", "condition", "present", "score", "accepted", "latency_ms"]
    with (output/"predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            csv_row = dict(row)
            if "query_category_ids" in csv_row:
                csv_row["query_category_ids"] = json.dumps(csv_row["query_category_ids"], ensure_ascii=False)
            writer.writerow(csv_row)
    temporary = output/"predictions.final.jsonl"
    with temporary.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
    temporary.replace(output/"predictions.jsonl")
    # Successful completion leaves one canonical calibrated JSONL. On a failure,
    # the partial file and completed=False summary remain recoverable evidence.
    (output/"predictions.partial.jsonl").unlink()
    summary["completed"] = True
    write_json(output/"summary.json", summary)
    return summary

