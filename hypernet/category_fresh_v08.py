"""Fresh photographs for retained categories, with verified source provenance.

After excluding v03, v04, v05 and v06 sources, v08 deliberately retains earlier
category assignments. Only photo availability decides the predeclared cohort
size fallback. Neither predictions nor query pixels enter the planner.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import math

import numpy as np

from .category_coco import _annotation_index, _support_annotation
from .category_data import validate_category_manifest
from .category_fresh_v05 import _read, _write_new, verify_prior_annotation_binding
from .data import file_sha256, load_rgb, pixel_sha256


def _fixed_split_plan(annotation_path, assignments, *, seed, category_count,
                      episodes_per_category, positive_queries, negative_queries,
                      min_support_area, min_support_edge, max_selection_attempts):
    """Select balanced subsets without changing any retained category's split."""
    _, names, images, labels, annotations = _annotation_index(annotation_path)
    pools = {split: sorted(c for c, s in assignments.items() if s == split)
             for split in ("dev", "test")}
    half = category_count // 2
    if any(len(pool) < half for pool in pools.values()):
        raise ValueError("Insufficient preferred categories in a fixed split")
    by_category = defaultdict(set)
    support_options = defaultdict(dict)
    for iid in sorted(images):
        for name in labels[iid] & assignments.keys():
            by_category[name].add(iid)
            box = _support_annotation(iid, name, names, annotations, images,
                                      min_support_area, min_support_edge)
            if box is not None:
                support_options[name][iid] = box
    rng = np.random.default_rng(seed)
    failure_counts = defaultdict(int)
    for attempt in range(max_selection_attempts):
        chosen = [pools[split][i] for split in ("dev", "test")
                  for i in rng.permutation(len(pools[split]))[:half]]
        split_names = {s: {c for c in chosen if assignments[c] == s}
                       for s in pools}
        valid_by_split = {
            s: {i for i in images if labels[i] & split_names[s]
                and not labels[i] & split_names["test" if s == "dev" else "dev"]}
            for s in pools}
        valid_by_category = {c: sorted(by_category[c] & valid_by_split[assignments[c]])
                             for c in chosen}
        support_candidates = {c: sorted(set(support_options[c]) & set(valid_by_category[c]))
                              for c in chosen}
        shortage = [c for c in chosen if len(support_candidates[c]) < episodes_per_category
                    or len(valid_by_category[c]) < episodes_per_category + positive_queries]
        if shortage:
            for c in shortage:
                failure_counts[c] += 1
            continue
        reserved = set()
        supports = {}
        for c in sorted(chosen, key=lambda c: (len(support_candidates[c]), c)):
            available = [i for i in support_candidates[c] if i not in reserved]
            if len(available) < episodes_per_category:
                break
            supports[c] = [available[i] for i in rng.choice(len(available), episodes_per_category, replace=False)]
            reserved.update(supports[c])
        if len(supports) != len(chosen):
            failure_counts["support_reservation"] += 1
            continue
        valid_by_category = {c: [i for i in ids if i not in reserved]
                             for c, ids in valid_by_category.items()}
        valid_by_split = {s: ids - reserved for s, ids in valid_by_split.items()}
        if any(len(valid_by_category[c]) < positive_queries or
               sum(c not in labels[i] for i in valid_by_split[assignments[c]]) < negative_queries
               for c in chosen):
            failure_counts["query_reservation"] += 1
            continue
        break
    else:
        raise ValueError(f"Fixed-split quotas failed after {max_selection_attempts} annotation-only attempts; shortages={dict(failure_counts)}")
    required = set()
    episodes = []
    for c in chosen:
        positive_pool = valid_by_category[c]
        negative_pool = sorted(i for i in valid_by_split[assignments[c]] if c not in labels[i])
        positives = [positive_pool[i] for i in rng.choice(len(positive_pool), positive_queries, replace=False)]
        negatives = [negative_pool[i] for i in rng.choice(len(negative_pool), negative_queries, replace=False)]
        for index, iid in enumerate(supports[c]):
            a = support_options[c][iid]
            episodes.append({"episode_id": "coco_" + hashlib.sha256(c.encode()).hexdigest()[:12] + f"_{index:02d}",
                             "category_id": c, "split": assignments[c], "support_image_id": int(iid),
                             "support_annotation_id": int(a["id"]), "support_bbox_xywh": list(a["bbox"]),
                             "positive_query_image_ids": [int(i) for i in positives],
                             "negative_query_image_ids": [int(i) for i in negatives]})
            required.update([iid, *positives, *negatives])
    return {"plan_version": "coco-category-plan-v1", "annotation_sha256": file_sha256(annotation_path),
            "seed": seed,
            "settings": {"max_categories": category_count, "dev_categories": half,
                         "episodes_per_category": episodes_per_category, "positive_queries": positive_queries,
                         "negative_queries": negative_queries, "exclude_categories": [],
                         "fixed_queries_across_supports": True, "min_support_area": min_support_area,
                         "min_support_edge": min_support_edge, "max_selection_attempts": max_selection_attempts,
                         "preserve_preferred_category_splits": True},
            "categories": [{"category_id": c, "split": assignments[c]} for c in chosen],
            "episodes": episodes,
            "required_images": [{k: images[i][k] for k in ("id", "file_name", "width", "height", "coco_url")
                                 if k in images[i]} for i in sorted(required)],
            "selection_audit": {"preferred_categories": len(assignments), "successful_selection_attempt": attempt + 1,
                                "support_source_photos": len(episodes), "required_source_photos": len(required),
                                "category_exposure_policy": "Exclude photos containing a selected opposite-split category.",
                                "support_policy": "Globally unique support photos; all support photos excluded from queries.",
                                "model_scores_used": False},
            "annotation_budget": {"support_photos_per_episode": 1, "support_boxes_per_episode": 1,
                                  "query_boxes_model_input": 0,
                                  "note": "Largest eligible noncrowd target box identifies the support subject; query boxes are evaluation only."}}


def plan_fresh_coco_categories_v08(annotation_path, prior_manifest_paths, filtered_annotation_path,
                                  plan_path, *, preferred_manifest_path=None, seed=101,
                                  category_counts=(20, 16, 12), episodes_per_category=3,
                                  positive_queries=10, negative_queries=10,
                                  min_support_area=4096, min_support_edge=32,
                                  max_selection_attempts=512):
    """Keep previous category splits while selecting genuinely new source files.

    New raw/decoded hashes must also pass the post-download audit. Category
    reuse is intentional and explicitly recorded; this is not a new-class test.
    """
    annotation_path = Path(annotation_path).resolve()
    filtered_annotation_path = Path(filtered_annotation_path).resolve()
    plan_path = Path(plan_path).resolve()
    priors = [Path(p).resolve() for p in prior_manifest_paths]
    if not priors:
        raise ValueError("At least one prior manifest is required")
    if (filtered_annotation_path == plan_path or
            {filtered_annotation_path, plan_path} & {annotation_path, *priors}):
        raise ValueError("Outputs must be distinct and cannot replace source evidence")
    if filtered_annotation_path.exists() or plan_path.exists():
        raise FileExistsError("Preserve existing evidence: fresh planning outputs already exist")
    if (not category_counts or any(type(n) is not int or n < 4 or n % 2 for n in category_counts)
            or list(category_counts) != sorted(set(category_counts), reverse=True)):
        raise ValueError("Fallback category counts must be distinct descending even integers >=4")
    for value in (episodes_per_category, positive_queries, negative_queries, max_selection_attempts):
        if type(value) is not int or value < 1:
            raise ValueError("Support/query/attempt quotas must be positive integers")
    if any(not math.isfinite(v) or v <= 0 for v in (min_support_area, min_support_edge)):
        raise ValueError("Support area/edge limits must be finite and positive")
    preferred_path = Path(preferred_manifest_path or priors[-1]).resolve()
    if preferred_path not in priors:
        raise ValueError("Preferred manifest must be among the excluded prior manifests")
    source = _read(annotation_path)
    original_sha = file_sha256(annotation_path)
    known = {r["id"]: r for r in source["images"]}
    known_categories = {r["name"] for r in source["categories"]}
    if len(known) != len(source["images"]):
        raise ValueError("Duplicate original source image IDs")
    old_ids, old_categories, old_raw, old_pixels = set(), set(), set(), set()
    prior_records = []
    assignments = None
    for path in priors:
        prior = _read(path)
        binding = verify_prior_annotation_binding(path, prior, source, original_sha)
        old_categories.update(r["category_id"] for r in prior["categories"])
        if path == preferred_path:
            assignments = {r["category_id"]: r["split"] for r in prior["categories"]}
            if len(assignments) != len(prior["categories"]):
                raise ValueError("Duplicate preferred category assignments")
        for row in prior["images"]:
            if type(row.get("coco_image_id")) is not int or row["coco_image_id"] not in known:
                raise ValueError("Prior images must retain valid original COCO image IDs")
            old_ids.add(row["coco_image_id"])
            for key, hashes in (("source_sha256", old_raw), ("source_pixel_sha256", old_pixels)):
                value = row.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"Missing or invalid prior {key}")
                hashes.add(value)
        prior_records.append({"path": str(path), "sha256": file_sha256(path), "annotation_binding": binding})
    if (not assignments or not set(assignments) <= known_categories or
            any(s not in {"dev", "test"} for s in assignments.values()) or
            not old_categories <= known_categories):
        raise ValueError("Prior categories/splits must use the original taxonomy")
    old_names = {known[i]["file_name"] for i in old_ids}
    removed = old_ids | {i for i, row in known.items() if row["file_name"] in old_names}
    retained = set(known) - removed
    filtered = {**source, "images": [r for r in source["images"] if r["id"] in retained],
                "annotations": [r for r in source["annotations"] if r["image_id"] in retained]}
    _write_new(filtered_annotation_path, filtered)
    attempts = []
    plan = None
    for count in category_counts:
        try:
            plan = _fixed_split_plan(filtered_annotation_path, assignments, seed=seed, category_count=count,
                                    episodes_per_category=episodes_per_category, positive_queries=positive_queries,
                                    negative_queries=negative_queries, min_support_area=min_support_area,
                                    min_support_edge=min_support_edge, max_selection_attempts=max_selection_attempts)
            attempts.append({"categories": count, "feasible": True})
            break
        except ValueError as error:
            attempts.append({"categories": count, "feasible": False, "reason": str(error)})
    if plan is None:
        raise ValueError(f"No predefined fixed-split cohort met source quotas: {attempts}")
    targets = {r["category_id"] for r in plan["categories"]}
    selected_ids = {r["id"] for r in plan["required_images"]}
    if selected_ids & removed:
        raise AssertionError("Fresh planner reused a prior source photograph")
    plan["freshness"] = {
        "original_annotation_path": str(annotation_path), "original_annotation_sha256": original_sha,
        "filtered_annotation_path": str(filtered_annotation_path),
        "filtered_annotation_sha256": file_sha256(filtered_annotation_path),
        "prior_manifests": prior_records, "preferred_manifest_path": str(preferred_path),
        "preferred_manifest_sha256": file_sha256(preferred_path),
        "category_policy": "Retain earlier category types and their DEV/TEST assignments; fresh photographs, not new types.",
        "preferred_category_assignments": assignments,
        "excluded_prior_source_ids": sorted(old_ids), "excluded_prior_source_filenames": sorted(old_names),
        "prior_source_sha256": sorted(old_raw), "prior_source_pixel_sha256": sorted(old_pixels),
        "retained_source_photos": len(retained), "retained_annotation_records": len(filtered["annotations"]),
        "annotation_preservation": "Every original image record, annotation, category and license for retained photos is preserved.",
        "selected_target_category_overlap": sorted(targets & old_categories),
        "dropped_preferred_categories": sorted(set(assignments) - targets), "selected_source_id_overlap": [],
        "predeclared_category_counts": list(category_counts), "cohort_size_attempts": attempts,
        "scores_used_for_selection": False,
        "limits": ["The selected category types have been used previously; this measures transfer to new photographs.",
                   "Distinct IDs and exact hashes do not establish different physical individuals, capture sessions, or absence of near-duplicates.",
                   "Raw and decoded source hashes must pass the post-download audit before evaluation."]}
    _write_new(plan_path, plan)
    return plan


def audit_fresh_category_manifest_v08(new_manifest_path, prior_manifest_paths, *, source_images_dir=None):
    """Check every source ID and hash; optionally verify cached originals too."""
    path = Path(new_manifest_path).resolve()
    fresh = _read(path)
    validation = validate_category_manifest(fresh, path.parent)
    prior_paths = [Path(p).resolve() for p in prior_manifest_paths]
    if not prior_paths:
        raise ValueError("At least one prior manifest is required")
    priors = [_read(p) for p in prior_paths]
    old_ids = {r["coco_image_id"] for m in priors for r in m["images"]}
    old_raw = {r["source_sha256"] for m in priors for r in m["images"]}
    old_pixels = {r["source_pixel_sha256"] for m in priors for r in m["images"]}
    old_categories = {r["category_id"] for m in priors for r in m["categories"]}
    new_ids = {r["coco_image_id"] for r in fresh["images"]}
    new_categories = {r["category_id"] for r in fresh["categories"]}
    overlaps = {"source_ids": sorted(new_ids & old_ids),
                "source_sha256": sorted({r["source_sha256"] for r in fresh["images"]} & old_raw),
                "source_pixel_sha256": sorted({r["source_pixel_sha256"] for r in fresh["images"]} & old_pixels)}
    if any(overlaps.values()):
        raise ValueError(f"Fresh-source overlap detected: {overlaps}")
    verified = 0
    if source_images_dir is not None:
        cache = Path(source_images_dir).resolve()
        seen = {}
        for manifest in [*priors, fresh]:
            for row in manifest["images"]:
                original = (cache / row["source_relative_path"]).resolve()
                if not original.is_relative_to(cache) or not original.is_file():
                    raise ValueError("Missing or unsafe original source photograph")
                if original not in seen:
                    seen[original] = (file_sha256(original), pixel_sha256(load_rgb(original)))
                    verified += 1
                if seen[original] != (row["source_sha256"], row["source_pixel_sha256"]):
                    raise ValueError(f"Original source hash mismatch: {original.name}")
    return {"valid": True, "new_manifest_sha256": file_sha256(path),
            "n_target_categories": len(new_categories), "n_source_photos": len(new_ids),
            "prior_target_categories": len(old_categories), "prior_source_photos": len(old_ids),
            "overlaps": overlaps, "reused_target_categories": sorted(new_categories & old_categories),
            "original_source_files_verified": verified, "stored_dataset_validation": validation,
            "prior_manifests": [{"path": str(p), "sha256": file_sha256(p)} for p in prior_paths],
            "limits": "Fresh photographs of reused category types; exact hashes do not exclude near-duplicates or prove physical-instance/capture independence."}
