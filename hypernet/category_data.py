"""One-shot category-presence episodes from independently stored source photographs.

Folder categories are evaluation labels. They do not identify physical subjects
or capture sessions. This schema intentionally does not reuse object_id equality.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image
from .data import load_rgb, pixel_sha256, file_sha256, _resolve

SCHEMA_VERSION = "category-pairs-v1"
PROTOCOL = "one_shot_category_presence"
DEFAULT_EXCLUSIONS = ("BACKGROUND_Google", "Faces_easy", "Faces", "cougar_face")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _thumbnail(path, size):
    image = Image.fromarray(load_rgb(path))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def load_category_manifest(path, validate=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["root"] = str(path.parent)
    if validate:
        manifest["validation"] = validate_category_manifest(manifest, path.parent)
    return manifest


def validate_category_manifest(manifest, root=None):
    """Check labels, separate category/instance/capture IDs, and content lineage.

    The builder reserves every support photo from every query pool, even across
    episodes. Validation enforces the same rule. Unknown instance/capture IDs
    must remain null; distinct files do not prove different physical subjects.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version={SCHEMA_VERSION!r}")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected protocol={PROTOCOL!r}")
    root = Path(root or manifest.get("root", ".")).resolve()
    pairs, images, categories = (manifest.get(k) for k in ("pairs", "images", "categories"))
    if not all(isinstance(v, list) and v for v in (pairs, images, categories)):
        raise ValueError("Nonempty pairs, images, and categories lists are required")
    if manifest.get("input_track") != "full_photo_no_query_annotation":
        raise ValueError("This schema requires the full_photo_no_query_annotation input track")

    category_splits = {}
    for item in categories:
        if not isinstance(item, dict):
            raise ValueError("Category records must be objects")
        cid = _text(item.get("category_id"), "category_id")
        split = item.get("split")
        if cid in category_splits or split not in {"dev", "test"}:
            raise ValueError("Unique categories with dev/test split are required")
        category_splits[cid] = split
    if any(sum(s == split for s in category_splits.values()) < 2 for split in ("dev", "test")):
        raise ValueError("At least two categories in each development/test split are required")

    by_image = {}; contents = defaultdict(set); identities = defaultdict(set); captures = defaultdict(set)
    paths = {}; seen_content = {}; original_content = {}
    for record in images:
        if not isinstance(record, dict):
            raise ValueError("Image records must be objects")
        iid = _text(record.get("image_id"), "image_id")
        cid = record.get("category_id")
        if iid in by_image or cid not in category_splits:
            raise ValueError("Unique image IDs with known categories are required")
        split = category_splits[cid]
        image_categories = record.get("category_ids", [cid])
        source_categories = record.get("source_category_ids", image_categories)
        for category_list in (image_categories, source_categories):
            if not isinstance(category_list, list) or not category_list or any(not isinstance(c, str) or not c for c in category_list) or len(set(category_list)) != len(category_list):
                raise ValueError("Image/source category_ids must be unique nonempty strings")
            if any(c in category_splits and category_splits[c] != split for c in category_list):
                raise ValueError("Cross-split category exposure in image/source labels")
        if cid not in image_categories:
            raise ValueError("Representative image category must be among its category_ids")
        if not set(image_categories).issubset(source_categories):
            raise ValueError("Stored image categories cannot add labels absent from source")
        path = _resolve(root, record.get("path"))
        if path in paths:
            raise ValueError("Two image IDs cannot alias the same path")
        paths[path] = iid
        raw = file_sha256(path); decoded = load_rgb(path); pixels = pixel_sha256(decoded)
        if record.get("sha256") != raw or record.get("pixel_sha256") != pixels:
            raise ValueError(f"Image hash mismatch: {record.get('path')}")
        source_raw = _sha(record.get("source_sha256"), "source_sha256")
        source_pixels = _sha(record.get("source_pixel_sha256"), "source_pixel_sha256")
        _text(record.get("source_record_id"), "source_record_id")
        for key in ("instance_id", "capture_id"):
            if key not in record or (record[key] is not None and (not isinstance(record[key], str) or not record[key].strip())):
                raise ValueError(f"{key} must be a nonempty string or explicit null")
        if record["instance_id"] is not None:
            identities[record["instance_id"]].add(split)
        if record["capture_id"] is not None:
            captures[record["capture_id"]].add(split)
        for key in (("raw", raw), ("pixels", pixels)):
            if key in seen_content:
                raise ValueError("Duplicate decoded/raw image content under different image IDs")
            seen_content[key] = iid
            contents[key].add(split)
        for key in (("source_raw", source_raw), ("source_pixels", source_pixels)):
            if key in original_content:
                raise ValueError("Duplicate source content under different image IDs")
            original_content[key] = iid
            contents[key].add(split)
        by_image[iid] = {**record, "_height": decoded.shape[0], "_width": decoded.shape[1]}

    seen_pairs = set(); seen_episode_queries = set()
    episode_info = {}; support_id_info = {}; supports = set(); queries = set()
    split_counts = Counter(); labels = Counter(); conditions = Counter(); episode_counts = Counter()
    positive_distinct_instances = 0; positive_unknown_instances = 0; source_records = defaultdict(set)
    for iid, record in by_image.items():
        source_records[record["source_record_id"]].add(iid)
    if any(len(v) > 1 for v in source_records.values()):
        raise ValueError("Source record IDs cannot alias different images")
    for row in pairs:
        required = {"id", "category_id", "episode_id", "support_id",
                    "reference", "query", "support_image_id", "query_image_id", "present", "split", "condition",
                    "support_instance_id", "query_instance_id", "support_capture_id", "query_capture_id"}
        if not isinstance(row, dict) or required - row.keys():
            raise ValueError(f"Pair missing required fields: {sorted(required - row.keys()) if isinstance(row,dict) else sorted(required)}")
        pid = _text(row["id"], "pair id")
        if pid in seen_pairs:
            raise ValueError("Duplicate pair ID")
        seen_pairs.add(pid)
        if type(row["present"]) is not bool:
            raise ValueError("present must be a JSON boolean")
        cid, qcid = row["category_id"], row.get("query_category_id")
        query_categories = row.get("query_category_ids", [qcid] if qcid is not None else [])
        if not isinstance(query_categories, list) or not query_categories or any(not isinstance(c, str) or not c for c in query_categories) or len(set(query_categories)) != len(query_categories):
            raise ValueError("query_category_ids must contain unique category strings")
        split = row["split"]
        if cid not in category_splits or (qcid is not None and qcid not in query_categories):
            raise ValueError("Unknown pair category")
        if split != category_splits[cid] or any(c in category_splits and split != category_splits[c] for c in query_categories):
            raise ValueError("Cross-split category leakage")
        if row["present"] != (cid in query_categories):
            raise ValueError("present disagrees with category labels")
        if any(k in row for k in ("object_id", "query_object_id")):
            raise ValueError("Do not conflate category IDs with old object_id semantics")
        sid, qid = row["support_image_id"], row["query_image_id"]
        if sid not in by_image or qid not in by_image:
            raise ValueError("Unknown support/query image ID")
        support, query = by_image[sid], by_image[qid]
        if sid == qid:
            raise ValueError("Support/query cannot be the same image")
        if set(query_categories) != set(query.get("category_ids", [query["category_id"]])):
            raise ValueError("Query category list disagrees with image registry")
        if cid not in support.get("category_ids", [support["category_id"]]):
            raise ValueError("Support does not contain its target category")
        for record, field, prefix in ((support, "reference", "support"), (query, "query", "query")):
            if row[field] != record["path"]:
                raise ValueError("Pair image path/category disagrees with image registry")
            for name in ("instance_id", "capture_id"):
                if row[prefix + "_" + name] != record[name]:
                    raise ValueError(f"Pair {prefix}_{name} disagrees with image registry")
            for suffix, key in (("_sha256", "sha256"), ("_pixel_sha256", "pixel_sha256")):
                if field + suffix in row and row[field + suffix] != record[key]:
                    raise ValueError("Pair hash disagrees with image registry")
        if support["capture_id"] is not None and support["capture_id"] == query["capture_id"]:
            raise ValueError("Support/query share a known capture session")
        episode = _text(row["episode_id"], "episode_id")
        support_id = _text(row["support_id"], "support_id")
        episode_key = (cid, split, sid, support_id)
        if episode in episode_info and episode_info[episode] != episode_key:
            raise ValueError("An episode must have exactly one support/category/split")
        episode_info[episode] = episode_key
        if support_id in support_id_info and support_id_info[support_id] != (cid, sid):
            raise ValueError("A support ID must name one reference/category")
        support_id_info[support_id] = (cid, sid)
        if (episode, qid) in seen_episode_queries:
            raise ValueError("Duplicate query within an episode")
        seen_episode_queries.add((episode, qid))
        supports.add(sid); queries.add(qid)
        if row.get("derived_from_reference") is True:
            raise ValueError("Category queries cannot be synthetic derivatives of support")
        if any(row.get(k) is not None for k in ("H_ref_to_query", "target_bbox", "reference_bbox", "query_bbox")):
            raise ValueError("Put any optional annotations in evaluation_annotations, not model input fields")
        if "evaluation_annotations" in row:
            annotation = row["evaluation_annotations"]
            if not isinstance(annotation, dict):
                raise ValueError("evaluation_annotations must be an object")
            if annotation.get("query_target_boxes") is not None:
                if not row["present"]:
                    raise ValueError("An absent category cannot have target boxes")
                for box in annotation["query_target_boxes"]:
                    arr = np.asarray(box, float)
                    if arr.shape != (4,) or not np.isfinite(arr).all() or arr[2] <= arr[0] or arr[3] <= arr[1]:
                        raise ValueError("Evaluation target boxes must be finite ordered xyxy")
                    if arr[0] < 0 or arr[1] < 0 or arr[2] > query["_width"] or arr[3] > query["_height"]:
                        raise ValueError("Evaluation target boxes must lie within the query image")
        if row["present"]:
            if support["instance_id"] is None or query["instance_id"] is None:
                positive_unknown_instances += 1
            elif support["instance_id"] != query["instance_id"]:
                positive_distinct_instances += 1
        split_counts[split] += 1; labels[str(row["present"]).lower()] += 1
        conditions[_text(row["condition"], "condition")] += 1
        episode_counts[(episode, row["present"])] += 1
    if supports & queries:
        raise ValueError("Support photos are reserved from every query pool across episodes")
    for episode in episode_info:
        if not episode_counts[(episode, True)] or not episode_counts[(episode, False)]:
            raise ValueError("Every episode requires present and absent queries")
    if manifest.get("settings", {}).get("fixed_queries_across_supports"):
        category_query_sets = defaultdict(list)
        for episode, info in episode_info.items():
            category_query_sets[info[0]].append({(r["query_image_id"], r["present"]) for r in pairs if r["episode_id"] == episode})
        if any(any(s != sets[0] for s in sets[1:]) for sets in category_query_sets.values()):
            raise ValueError("Query sets must be identical across support episodes of each category")
    for name, groups in (("instance", identities), ("capture", captures), ("image content", contents)):
        if any(len(splits) > 1 for splits in groups.values()):
            raise ValueError(f"Cross-split {name} leakage")
    used = supports | queries
    if used != set(by_image):
        raise ValueError("Image registry must contain exactly the used images")
    return {"valid": True, "n_pairs": len(pairs), "n_categories": len(category_splits),
            "n_episodes": len(episode_info), "n_support_images": len(supports), "n_images": len(by_image),
            "splits": dict(split_counts), "labels": dict(labels), "conditions": dict(conditions),
            "category_splits": {s: sorted(c for c, v in category_splits.items() if v == s) for s in ("dev", "test")},
            "support_query_disjoint": True, "category_disjoint_dev_test": True,
            "positive_pairs_with_verified_different_instance_ids": positive_distinct_instances,
            "positive_pairs_with_unknown_instance_ids": positive_unknown_instances,
            "limitation": "Distinct source files and category labels do not verify physical-instance or capture-session independence. Exact content checks do not detect all near duplicates."}


def build_category_dataset(source_dir, out_dir, *, seed=53, max_categories=24,
                           dev_categories=None, episodes_per_category=3,
                           positive_queries=10, negative_queries=10, size=256,
                           exclude_categories=DEFAULT_EXCLUSIONS):
    """Build portable category-presence episodes; never download or overwrite.

    Support photographs are reserved across all episodes. Queries are original,
    separately stored source photographs resized without synthetic transformations.
    Source folder labels provide category truth, not physical-instance identities.
    """
    source_dir, out_dir = Path(source_dir).resolve(), Path(out_dir).resolve()
    if (source_dir / "101_ObjectCategories").is_dir():
        source_dir = source_dir / "101_ObjectCategories"
    for name, value, minimum in (("max_categories", max_categories, 4), ("episodes_per_category", episodes_per_category, 1),
                                 ("positive_queries", positive_queries, 1), ("negative_queries", negative_queries, 1), ("size", size, 16)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Category output directory must be empty")
    if out_dir == source_dir or source_dir.is_relative_to(out_dir) or out_dir.is_relative_to(source_dir):
        raise ValueError("Source and output directories must be separate")
    rng = np.random.default_rng(seed)
    seen_raw, seen_pixels, seen_resized = set(), set(), set()
    category_records = {}; skipped = Counter()
    for folder in sorted(source_dir.iterdir()):
        if not folder.is_dir() or folder.name.startswith(".") or folder.name in set(exclude_categories):
            continue
        records = []
        for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
            try:
                raw_hash = file_sha256(path)
                rgb = load_rgb(path); original_hash = pixel_sha256(rgb)
                thumb = Image.fromarray(rgb); thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
                resized_hash = pixel_sha256(np.asarray(thumb, dtype=np.uint8))
            except (OSError, ValueError):
                skipped["unreadable_images"] += 1
                continue
            if raw_hash in seen_raw or original_hash in seen_pixels or resized_hash in seen_resized:
                skipped["duplicate_source_or_resized_content"] += 1
                continue
            seen_raw.add(raw_hash); seen_pixels.add(original_hash); seen_resized.add(resized_hash)
            records.append({"source": path, "source_relative_path": path.relative_to(source_dir).as_posix(),
                            "source_sha256": raw_hash, "source_pixel_sha256": original_hash,
                            "resized_pixel_sha256": resized_hash, "category_id": folder.name})
        if len(records) >= episodes_per_category + positive_queries:
            category_records[folder.name] = records
        else:
            skipped["categories_with_too_few_unique_photos"] += 1
    eligible = sorted(category_records)
    if len(eligible) < max_categories:
        raise ValueError(f"Need {max_categories} eligible categories, found {len(eligible)}")
    chosen = [eligible[i] for i in rng.permutation(len(eligible))[:max_categories]]
    ndev = max_categories // 2 if dev_categories is None else dev_categories
    if type(ndev) is not int or not 2 <= ndev <= max_categories - 2:
        raise ValueError("dev_categories must leave at least two categories in each split")
    assignments = {cid: "dev" if i < ndev else "test" for i, cid in enumerate(chosen)}
    selected_supports = {}; query_pools = {}
    for cid in chosen:
        records = category_records[cid]
        perm = rng.permutation(len(records))
        selected_supports[cid] = [records[i] for i in perm[:episodes_per_category]]
        query_pools[cid] = [records[i] for i in perm[episodes_per_category:]]
    episodes = []
    used = {}
    for cid in chosen:
        negatives = [r for other in chosen if other != cid and assignments[other] == assignments[cid] for r in query_pools[other]]
        if len(negatives) < negative_queries:
            raise ValueError(f"Not enough negative query photographs in {assignments[cid]} split")
        positives = [query_pools[cid][i] for i in rng.choice(len(query_pools[cid]), positive_queries, replace=False)]
        absent = [negatives[i] for i in rng.choice(len(negatives), negative_queries, replace=False)]
        for ep, support in enumerate(selected_supports[cid]):
            episodes.append((cid, ep, support, positives, absent))
            for r in [support, *positives, *absent]:
                used[r["source_relative_path"]] = r
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir()
    image_records = []; registry = {}
    for source_path, r in sorted(used.items()):
        relative = "images/" + r["source_pixel_sha256"] + ".png"
        rgb = _thumbnail(r["source"], size)
        Image.fromarray(rgb).save(out_dir / relative)
        record = {"image_id": "image_" + r["source_pixel_sha256"],
                  "category_id": r["category_id"], "category_ids": [r["category_id"]], "path": relative,
                  "sha256": file_sha256(out_dir / relative), "pixel_sha256": pixel_sha256(rgb),
                  "source_relative_path": source_path, "source_sha256": r["source_sha256"],
                  "source_pixel_sha256": r["source_pixel_sha256"],
                  "source_record_id": source_path, "instance_id": None, "capture_id": None}
        registry[source_path] = record; image_records.append(record)
    pairs = []
    for cid, ep, source, positives, absent in episodes:
        support = registry[source["source_relative_path"]]
        episode = "episode_" + hashlib.sha256(cid.encode()).hexdigest()[:12] + f"_{ep:02d}"
        for present, choices in ((True, positives), (False, absent)):
            for i, q in enumerate(choices):
                query = registry[q["source_relative_path"]]
                pairs.append({"id": f"{episode}_{'present' if present else 'absent'}_{i:03d}",
                              "category_id": cid, "query_category_id": query["category_id"], "query_category_ids": [query["category_id"]],
                              "episode_id": episode, "support_id": support["image_id"],
                              "reference": support["path"], "query": query["path"],
                              "support_image_id": support["image_id"], "query_image_id": query["image_id"],
                              "reference_sha256": support["sha256"], "query_sha256": query["sha256"],
                              "reference_pixel_sha256": support["pixel_sha256"], "query_pixel_sha256": query["pixel_sha256"],
                              "support_instance_id": None, "query_instance_id": None,
                              "support_capture_id": None, "query_capture_id": None,
                              "present": present, "split": assignments[cid], "condition": "different_photo"})
    manifest = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
                "input_track": "full_photo_no_query_annotation", "seed": seed,
                "category_taxonomy": "Original source-folder labels after the listed exclusions; no model receives label text.",
                "categories": [{"category_id": cid, "split": assignments[cid]} for cid in chosen],
                "images": image_records, "pairs": pairs,
                "settings": {"max_categories": max_categories, "dev_categories": ndev,
                             "episodes_per_category": episodes_per_category, "positive_queries": positive_queries,
                             "negative_queries": negative_queries, "max_image_side": size,
                             "fixed_queries_across_supports": True,
                             "exclude_categories": list(exclude_categories)},
                "selection_audit": {"eligible_categories": len(eligible), "skipped": dict(skipped),
                                    "support_query_pool_policy": "All selected support photos excluded from all query pools across episodes.",
                                    "duplicate_policy": "Global raw source, decoded source pixels, and resized pixels deduplicated before category selection."},
                "notes": ["Each episode receives exactly one support photograph and no query annotations.",
                          "Category labels define presence. Another physical instance of the same category is positive, not an identity mismatch.",
                          "Source photos are resized to a maximum side without cropping, padding, synthetic warps, or compositing.",
                          "Categories are disjoint between development and test; this does not audit external pretraining overlap.",
                          "Queries can repeat across support episodes, so episodes are correlated; uncertainty should group by category."],
                "limitations": ["Source-folder labels are not exhaustive multi-object annotations. A nominal other-category image may contain an incidental target; annotation review is needed.",
                                "Physical instance and capture-session identities are unknown. Different files are not proof of different physical subjects or independent capture sessions.",
                                "Exact duplicate checks cannot rule out near duplicates, crops, or related source photographs.",
                                "Source datasets with centered subjects do not establish robust category localization in clutter."]}
    manifest["validation"] = validate_category_manifest(manifest, out_dir)
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target

