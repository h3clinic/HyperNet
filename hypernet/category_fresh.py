"""Fresh COCO category planning without rewriting category labels or annotations.

This wrapper excludes earlier target categories and previously used source
photographs, then delegates selection to the frozen category planner. All
annotations on every retained photo remain present, including incidental classes.
"""
from __future__ import annotations

from pathlib import Path
import json

from .category_coco import plan_coco_category_dataset
from .data import file_sha256


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Preserve existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def plan_fresh_coco_categories(annotation_path, prior_manifest_paths,
                              filtered_annotation_path, plan_path, *, seed=61,
                              category_counts=(24, 20, 16, 12),
                              episodes_per_category=3, positive_queries=10,
                              negative_queries=10, exclude_categories=("person",),
                              min_support_area=4096, min_support_edge=32,
                              max_selection_attempts=128):
    """Create a fresh-source planning annotation and a selected category plan.

    Fallback cohort sizes are tried in the supplied order using only annotation
    availability, never scores. Exactly half of each chosen cohort calibrates
    cutoffs; the remainder is final test categories. Reusing old development
    data to design a method is separate from calibration on the new development
    partition. The return value is directly accepted by build_coco_category_dataset
    with filtered_annotation_path.

    Different COCO IDs do not prove different physical individuals. Freshness of
    source bytes/decoded pixels must additionally be checked after downloading.
    """
    annotation_path = Path(annotation_path).resolve()
    filtered_annotation_path = Path(filtered_annotation_path).resolve()
    plan_path = Path(plan_path).resolve()
    if filtered_annotation_path == annotation_path or plan_path == annotation_path:
        raise ValueError("Output paths must not replace the source annotations")
    if filtered_annotation_path == plan_path:
        raise ValueError("Filtered annotations and plan need distinct paths")
    if filtered_annotation_path.exists() or plan_path.exists():
        raise FileExistsError("Fresh planning outputs must not already exist")
    prior_paths = [Path(p).resolve() for p in prior_manifest_paths]
    if not prior_paths:
        raise ValueError("At least one prior manifest is required")
    if any(p in {filtered_annotation_path, plan_path} for p in prior_paths):
        raise ValueError("Fresh outputs cannot replace prior manifests")
    if not category_counts or any(type(n) is not int or n < 4 or n % 2 for n in category_counts):
        raise ValueError("Fallback category counts must be positive even integers >=4")
    if len(set(category_counts)) != len(category_counts):
        raise ValueError("Fallback category counts must be unique")

    source = _read(annotation_path)
    original_sha = file_sha256(annotation_path)
    known_categories = {r["name"] for r in source["categories"]}
    known_images = {r["id"]: r for r in source["images"]}
    if len(known_images) != len(source["images"]):
        raise ValueError("COCO source contains duplicate image IDs")
    old_targets = set()
    old_sources = set()
    old_filenames = set()
    prior_hashes = set()
    prior_pixels = set()
    prior_records = []
    for p in prior_paths:
        prior = _read(p)
        if prior.get("annotation_sha256") not in (None, original_sha):
            raise ValueError("Prior manifest is bound to a different original annotation file")
        old_targets.update(r["category_id"] for r in prior["categories"])
        for image in prior["images"]:
            iid = image.get("coco_image_id")
            if type(iid) is not int or iid not in known_images:
                raise ValueError("Prior images must retain valid original COCO image IDs")
            old_sources.add(iid)
            old_filenames.add(known_images[iid]["file_name"])
            if image.get("source_sha256"):
                prior_hashes.add(image["source_sha256"])
            if image.get("source_pixel_sha256"):
                prior_pixels.add(image["source_pixel_sha256"])
        prior_records.append({"path": str(p), "sha256": file_sha256(p)})
    if not old_targets <= known_categories:
        raise ValueError("Prior target categories must use the original COCO taxonomy")

    excluded_ids = old_sources | {i for i, im in known_images.items() if im["file_name"] in old_filenames}
    retained = set(known_images) - excluded_ids
    # Keep complete annotation records and the entire category taxonomy. Never
    # drop a category annotation to make an absent label or an easier split.
    filtered = {**source,
                "images": [r for r in source["images"] if r["id"] in retained],
                "annotations": [r for r in source["annotations"] if r["image_id"] in retained]}
    _write_new(filtered_annotation_path, filtered)
    excluded_targets = sorted(old_targets | set(exclude_categories))
    attempts = []
    selected = None
    for n in category_counts:
        try:
            selected = plan_coco_category_dataset(
                filtered_annotation_path, seed=seed, max_categories=n,
                dev_categories=n//2, episodes_per_category=episodes_per_category,
                positive_queries=positive_queries, negative_queries=negative_queries,
                exclude_categories=excluded_targets, min_support_area=min_support_area,
                min_support_edge=min_support_edge, max_selection_attempts=max_selection_attempts)
            attempts.append({"categories": n, "feasible": True})
            break
        except ValueError as exc:
            attempts.append({"categories": n, "feasible": False, "reason": str(exc)})
    if selected is None:
        raise ValueError(f"No predefined fresh cohort met source quotas: {attempts}")
    new_targets = {r["category_id"] for r in selected["categories"]}
    new_sources = {r["id"] for r in selected["required_images"]}
    if new_targets & old_targets or new_sources & excluded_ids:
        raise AssertionError("Fresh planner reused a prior target or photo")

    prior_incidental_categories = {a["category_id"] for a in source["annotations"] if a["image_id"] in old_sources}
    names = {c["id"]: c["name"] for c in source["categories"]}
    prior_incidental_names = {names[c] for c in prior_incidental_categories}
    selected["freshness"] = {
        "original_annotation_path": str(annotation_path),
        "original_annotation_sha256": original_sha,
        "filtered_annotation_path": str(filtered_annotation_path),
        "filtered_annotation_sha256": file_sha256(filtered_annotation_path),
        "prior_manifests": prior_records,
        "excluded_prior_target_categories": sorted(old_targets),
        "excluded_prior_source_ids": sorted(old_sources),
        "excluded_prior_source_filenames": sorted(old_filenames),
        "prior_source_sha256": sorted(prior_hashes),
        "prior_source_pixel_sha256": sorted(prior_pixels),
        "retained_source_photos": len(retained),
        "retained_annotation_records": len(filtered["annotations"]),
        "annotation_preservation": "Every annotation on retained source photos and the original category taxonomy are preserved.",
        "selected_target_category_overlap": [],
        "selected_source_id_overlap": [],
        "selected_targets_previously_incidental_in_prior_photos": sorted(new_targets & prior_incidental_names),
        "cohort_size_attempts": attempts,
        "scores_used_for_selection": False,
        "limits": [
            "New target categories may have appeared incidentally in earlier multi-object photos.",
            "Distinct source IDs do not prove different physical individuals or capture sessions.",
            "Raw and decoded source hashes must be checked after downloading; this plan only proves ID/category separation.",
        ],
    }
    _write_new(plan_path, selected)
    return selected


def audit_fresh_category_manifest(new_manifest_path, prior_manifest_paths):
    """Check source ID, raw bytes and decoded-pixel freshness after data import."""
    new_path = Path(new_manifest_path)
    fresh = _read(new_path)
    old_categories, old_ids, old_raw, old_pixels = set(), set(), set(), set()
    for prior_path in prior_manifest_paths:
        prior = _read(prior_path)
        old_categories.update(c["category_id"] for c in prior["categories"])
        for image in prior["images"]:
            old_ids.add(image["coco_image_id"])
            old_raw.add(image["source_sha256"])
            old_pixels.add(image["source_pixel_sha256"])
    new_categories = {c["category_id"] for c in fresh["categories"]}
    new_ids = {r["coco_image_id"] for r in fresh["images"]}
    new_raw = {r["source_sha256"] for r in fresh["images"]}
    new_pixels = {r["source_pixel_sha256"] for r in fresh["images"]}
    overlaps = {
        "target_categories": sorted(new_categories & old_categories),
        "source_ids": sorted(new_ids & old_ids),
        "source_sha256": sorted(new_raw & old_raw),
        "source_pixel_sha256": sorted(new_pixels & old_pixels),
    }
    if any(overlaps.values()):
        raise ValueError(f"Fresh-category/source overlap detected: {overlaps}")
    return {"valid": True, "new_manifest_sha256": file_sha256(new_path),
            "n_target_categories": len(new_categories), "n_source_photos": len(new_ids),
            "prior_target_categories": len(old_categories), "prior_source_photos": len(old_ids),
            "overlaps": overlaps,
            "limits": "Hash disjointness does not establish physical-instance uniqueness or exclude near-duplicate photographs."}

