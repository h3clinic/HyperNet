"""Optional evaluation-tool bridges; exporting JSON needs no FiftyOne install.

The export represents binary reference/query decisions. Similarity scores stay
separate from Classification.confidence because they are not probabilities.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math
import os
from .data import load_manifest

EXPORT_VERSION = "hypernet-fiftyone-pairs-v1"


def _relative(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError as exc:
        raise ValueError("Export and dataset must be on the same drive for portable relative paths") from exc


def export_fiftyone_json(predictions_path, manifest_path, output_path, model):
    """Export one fully evaluated model's pair decisions to portable JSON.

    Reads the runner's ``predictions.jsonl``. Each selected record must contain
    an explicit boolean ``accepted`` produced after development calibration.
    No decision is guessed from a score, and no missing test labels are inferred.
    All manifest pair IDs must appear exactly once for the selected model.
    The output uses relative file paths and requires the associated image tree.
    """
    predictions_path, manifest_path, output_path = map(lambda p: Path(p).resolve(),
                                                     (predictions_path, manifest_path, output_path))
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty model alias")
    if output_path.exists():
        raise FileExistsError("Use a fresh export filename to preserve earlier results")
    manifest = load_manifest(manifest_path)
    root = Path(manifest["root"])
    expected = {pair["id"]: pair for pair in manifest["pairs"]}
    rows = {}; available = set()
    with predictions_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON prediction at line {line_number}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("model"), str):
                raise ValueError(f"Prediction line {line_number} needs a model alias")
            available.add(row["model"])
            if row["model"] != model:
                continue
            pair_id = row.get("id")
            if pair_id not in expected:
                raise ValueError(f"Unknown manifest pair ID: {pair_id!r}")
            if pair_id in rows:
                raise ValueError(f"Duplicate prediction for pair: {pair_id}")
            if type(row.get("accepted")) is not bool:
                raise ValueError(f"Pair {pair_id} needs an explicit boolean accepted decision; finish calibration first")
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Pair {pair_id} needs a finite numeric score")
            pair = expected[pair_id]
            for key in ("reference", "query", "present", "object_id", "query_object_id", "split", "condition"):
                if key in row and key in pair and row[key] != pair[key]:
                    raise ValueError(f"Prediction/manifest mismatch for {pair_id}: {key}")
            rows[pair_id] = row
    if model not in available:
        raise ValueError(f"Unknown model {model!r}; available models: {sorted(available)}")
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise ValueError(f"Incomplete model predictions: {len(missing)} manifest pairs missing, including {missing[:3]}")
    samples = []
    for pair in manifest["pairs"]:
        row = rows[pair["id"]]
        present, accepted = pair["present"], row["accepted"]
        samples.append({"pair_id": pair["id"],
                        "filepath": _relative(root / pair["query"], output_path.parent),
                        "reference_filepath": _relative(root / pair["reference"], output_path.parent),
                        "ground_truth": {"label": "present" if present else "absent"},
                        "prediction": {"label": "present" if accepted else "absent"},
                        "similarity_score": float(row["score"]), "accepted": accepted,
                        "present": present, "correct": present == accepted,
                        "model": model, "object_id": pair["object_id"],
                        "query_object_id": pair.get("query_object_id"),
                        "condition": pair["condition"], "split": pair["split"],
                        "tags": [pair["split"], pair["condition"], model]})
    payload = {"format": EXPORT_VERSION, "model": model, "protocol": manifest.get("protocol"),
               "manifest_path": _relative(manifest_path, output_path.parent),
               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
               "path_base": "Paths are relative to this export JSON file; retain the linked image tree.",
               "score_semantics": "Uncalibrated similarity, not probability. accepted is the runner's development-calibrated decision.",
               "task": "Binary reference-conditioned pair verification, not object detection",
               "source_dataset_notes": manifest.get("notes", manifest.get("description")),
               "samples": samples}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    return output_path


def _read_export(export_path):
    export_path = Path(export_path).resolve()
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    if payload.get("format") != EXPORT_VERSION or not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError(f"Expected a nonempty {EXPORT_VERSION} export")
    for sample in payload["samples"]:
        for field in ("filepath", "reference_filepath"):
            relative = sample.get(field)
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ":" in relative or relative.startswith("\\"):
                raise ValueError(f"Expected portable relative {field}")
            resolved = (export_path.parent / relative).resolve()
            if not resolved.is_file():
                raise ValueError(f"Linked image is missing: {relative}")
    return payload


def build_fiftyone_dataset(export_path, name=None):
    """Create a local FiftyOne dataset on explicit invocation, without launching UI.

    FiftyOne is imported lazily. Its local database service may start when a
    dataset is created. Nothing is uploaded; no browser/App session is launched.
    A duplicate dataset name raises rather than overwriting an existing dataset.
    """
    export_path = Path(export_path).resolve()
    payload = _read_export(export_path)
    try:
        import fiftyone as fo
    except ImportError as exc:
        raise ImportError("Optional bridge requires FiftyOne; JSON export itself does not") from exc
    dataset = fo.Dataset(name=name) if name is not None else fo.Dataset()
    dataset.info = {"hypernet_export_format": EXPORT_VERSION, "model": payload["model"],
                    "protocol": payload.get("protocol"), "task": payload["task"],
                    "score_semantics": payload["score_semantics"],
                    "manifest_sha256": payload["manifest_sha256"]}
    samples = []
    for row in payload["samples"]:
        sample = fo.Sample(filepath=str((export_path.parent / row["filepath"]).resolve()), tags=row["tags"])
        sample["reference_filepath"] = str((export_path.parent / row["reference_filepath"]).resolve())
        sample["ground_truth"] = fo.Classification(label=row["ground_truth"]["label"])
        sample["prediction"] = fo.Classification(label=row["prediction"]["label"])
        for field in ("pair_id", "similarity_score", "accepted", "present", "correct", "model",
                      "object_id", "query_object_id", "condition", "split"):
            sample[field] = row[field]
        samples.append(sample)
    dataset.add_samples(samples)
    dataset.save()
    return dataset
