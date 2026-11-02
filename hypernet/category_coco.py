"""Deterministic COCO category-presence planning and portable dataset import.

Planning reads annotations only, enabling the caller to fetch exactly the required
photographs. No function downloads images, and no query boxes enter model input.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from PIL import Image
from .category_data import SCHEMA_VERSION, PROTOCOL, validate_category_manifest
from .data import load_rgb, file_sha256, pixel_sha256


def _annotation_index(annotation_path):
    source = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"] for c in source["categories"]}
    if len(set(categories.values())) != len(categories):
        raise ValueError("COCO category names must be unique")
    images = {i["id"]: i for i in source["images"]}
    labels = defaultdict(set); annotations = defaultdict(list)
    for annotation in source["annotations"]:
        iid, cid = annotation["image_id"], annotation["category_id"]
        if iid not in images or cid not in categories:
            raise ValueError("Annotation references unknown image/category")
        labels[iid].add(categories[cid])
        annotations[iid].append(annotation)
    return source, categories, images, labels, annotations


def _support_annotation(iid, name, category_names, annotations, images, min_area, min_edge):
    choices = []
    width, height = images[iid]["width"], images[iid]["height"]
    for a in annotations[iid]:
        if category_names[a["category_id"]] != name or a.get("iscrowd", 0):
            continue
        box = np.asarray(a.get("bbox"), float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x, y, w, h = box
        x0, y0, x1, y1 = max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)
        if min(x1-x0, y1-y0) < min_edge or (x1-x0)*(y1-y0) < min_area:
            continue
        choices.append(((x1-x0)*(y1-y0), int(a["id"]), a))
    return max(choices, key=lambda item: (item[0], -item[1]))[2] if choices else None


def plan_coco_category_dataset(annotation_path, *, seed=53, max_categories=24, dev_categories=None,
                               episodes_per_category=3, positive_queries=10, negative_queries=10,
                               exclude_categories=("person",), min_support_area=4096,
                               min_support_edge=32, max_selection_attempts=128):
    """Return JSON-serializable image selection, fixed before model evaluation.

    Every source photo contains selected categories from only one split. All
    support source photos are globally distinct and excluded from every query.
    Query truth uses all annotated categories, including crowd annotations.
    """
    for name, value, minimum in (("max_categories", max_categories, 4),
                                ("episodes_per_category", episodes_per_category, 1),
                                ("positive_queries", positive_queries, 1),
                                ("negative_queries", negative_queries, 1),
                                ("max_selection_attempts", max_selection_attempts, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(min_support_area) or min_support_area <= 0 or not math.isfinite(min_support_edge) or min_support_edge <= 0:
        raise ValueError("Support area and edge limits must be finite positive values")
    ndev = max_categories // 2 if dev_categories is None else dev_categories
    if type(ndev) is not int or not 2 <= ndev <= max_categories-2:
        raise ValueError("At least two categories per split are required")
    _, category_names, images, labels, annotations = _annotation_index(annotation_path)
    by_category = defaultdict(set); support_options = defaultdict(dict)
    for iid in sorted(images):
        for name in labels[iid]:
            if name in exclude_categories:
                continue
            by_category[name].add(iid)
            annotation = _support_annotation(iid, name, category_names, annotations, images, min_support_area, min_support_edge)
            if annotation is not None:
                support_options[name][iid] = annotation
    eligible = sorted(name for name in by_category
                      if len(by_category[name]) >= episodes_per_category + positive_queries
                      and len(support_options[name]) >= episodes_per_category)
    if len(eligible) < max_categories:
        raise ValueError(f"Only {len(eligible)} categories meet the fixed source/support quotas; requested {max_categories}")
    rng = np.random.default_rng(seed)
    chosen = assignments = support_records = valid_by_category = valid_by_split = None
    for attempt in range(max_selection_attempts):
        chosen = [eligible[i] for i in rng.permutation(len(eligible))[:max_categories]]
        assignments = {name: "dev" if i < ndev else "test" for i, name in enumerate(chosen)}
        split_labels = {split: {name for name in chosen if assignments[name] == split} for split in ("dev", "test")}
        valid_by_split = {}
        for split in ("dev", "test"):
            other = split_labels["test" if split == "dev" else "dev"]
            valid_by_split[split] = {iid for iid in images if labels[iid] & split_labels[split] and not labels[iid] & other}
        valid_by_category = {name: sorted(by_category[name] & valid_by_split[assignments[name]]) for name in chosen}
        candidates = {name: sorted(set(support_options[name]) & set(valid_by_category[name])) for name in chosen}
        if any(len(candidates[name]) < episodes_per_category or len(valid_by_category[name]) < episodes_per_category+positive_queries for name in chosen):
            continue
        support_records = {}; reserved = set()
        # Allocate rarest support pools first, preserving unique source photos.
        for name in sorted(chosen, key=lambda n: (len(candidates[n]), n)):
            available = [iid for iid in candidates[name] if iid not in reserved]
            if len(available) < episodes_per_category:
                break
            selected = [available[i] for i in rng.choice(len(available), episodes_per_category, replace=False)]
            support_records[name] = selected; reserved.update(selected)
        if len(support_records) != len(chosen):
            continue
        valid_by_category = {name: [iid for iid in valid_by_category[name] if iid not in reserved] for name in chosen}
        valid_by_split = {split: valid_by_split[split]-reserved for split in valid_by_split}
        if any(len(valid_by_category[name]) < positive_queries or
               len([iid for iid in valid_by_split[assignments[name]] if name not in labels[iid]]) < negative_queries for name in chosen):
            continue
        break
    else:
        raise ValueError(f"No category-disjoint photo selection met the fixed quotas in {max_selection_attempts} attempts")
    episodes = []; required = set()
    for name in chosen:
        negatives = sorted(iid for iid in valid_by_split[assignments[name]] if name not in labels[iid])
        positive_pool = valid_by_category[name]
        pos = [positive_pool[i] for i in rng.choice(len(positive_pool), positive_queries, replace=False)]
        neg = [negatives[i] for i in rng.choice(len(negatives), negative_queries, replace=False)]
        for ep, support_iid in enumerate(support_records[name]):
            a = support_options[name][support_iid]
            episode = {"episode_id": "coco_"+hashlib.sha256(name.encode()).hexdigest()[:12]+f"_{ep:02d}",
                       "category_id": name, "split": assignments[name], "support_image_id": int(support_iid),
                       "support_annotation_id": int(a["id"]), "support_bbox_xywh": list(a["bbox"]),
                       "positive_query_image_ids": [int(i) for i in pos], "negative_query_image_ids": [int(i) for i in neg]}
            episodes.append(episode); required.update([support_iid, *pos, *neg])
    return {"plan_version": "coco-category-plan-v1", "annotation_sha256": file_sha256(annotation_path),
            "seed": seed, "settings": {"max_categories": max_categories, "dev_categories": ndev,
            "episodes_per_category": episodes_per_category, "positive_queries": positive_queries,
            "negative_queries": negative_queries, "exclude_categories": list(exclude_categories),
            "fixed_queries_across_supports": True,
            "min_support_area": min_support_area, "min_support_edge": min_support_edge,
            "max_selection_attempts": max_selection_attempts},
            "categories": [{"category_id": name, "split": assignments[name]} for name in chosen],
            "episodes": episodes,
            "required_images": [{k: images[iid][k] for k in ("id", "file_name", "width", "height", "coco_url") if k in images[iid]}
                                for iid in sorted(required)],
            "selection_audit": {"eligible_categories": len(eligible), "successful_selection_attempt": attempt+1,
                                "support_source_photos": len(episodes), "required_source_photos": len(required),
                                "category_exposure_policy": "Exclude every photo containing a selected category assigned to the opposite split.",
                                "support_policy": "Globally unique source photos; all support source photos excluded from all queries.",
                                "model_scores_used": False},
            "annotation_budget": {"support_photos_per_episode": 1, "support_boxes_per_episode": 1,
                                  "query_boxes_model_input": 0,
                                  "note": "Largest eligible noncrowd target box identifies the support subject; query boxes are evaluation only."}}


def _resize(rgb, size):
    im = Image.fromarray(rgb); im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def build_coco_category_dataset(annotation_path, images_dir, out_dir, *, plan=None, size=384, support_padding=.05,
                                seed=53, max_categories=24, dev_categories=None, episodes_per_category=3,
                                positive_queries=10, negative_queries=10, exclude_categories=("person",),
                                min_support_area=4096, min_support_edge=32, max_selection_attempts=128):
    if type(size) is not int or size < 16:
        raise ValueError("size must be an integer >=16")
    if not math.isfinite(support_padding) or not 0 <= support_padding <= .5:
        raise ValueError("support_padding must be finite in [0,.5]")
    images_dir, out_dir = Path(images_dir).resolve(), Path(out_dir).resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError("COCO category output directory must be empty")
    if out_dir == images_dir or out_dir.is_relative_to(images_dir) or images_dir.is_relative_to(out_dir):
        raise ValueError("Source and output directories must be separate")
    if plan is None:
        plan = plan_coco_category_dataset(annotation_path, seed=seed, max_categories=max_categories,
               dev_categories=dev_categories, episodes_per_category=episodes_per_category,
               positive_queries=positive_queries, negative_queries=negative_queries,
               exclude_categories=exclude_categories, min_support_area=min_support_area,
               min_support_edge=min_support_edge, max_selection_attempts=max_selection_attempts)
    elif isinstance(plan, (str, Path)):
        plan = json.loads(Path(plan).read_text(encoding="utf-8"))
    if plan.get("plan_version") != "coco-category-plan-v1" or plan.get("annotation_sha256") != file_sha256(annotation_path):
        raise ValueError("COCO plan does not match annotation source")
    _, category_names, source_images, labels, annotations = _annotation_index(annotation_path)
    assignments = {x["category_id"]: x["split"] for x in plan["categories"]}
    supports = {ep["support_image_id"]: ep for ep in plan["episodes"]}
    if len(supports) != len(plan["episodes"]):
        raise ValueError("Every support episode must use a distinct source photograph")
    query_ids = {iid for ep in plan["episodes"] for k in ("positive_query_image_ids", "negative_query_image_ids") for iid in ep[k]}
    if set(supports) & query_ids:
        raise ValueError("Support source photographs cannot appear among queries")
    required_ids = set(supports) | query_ids
    if required_ids != {r["id"] for r in plan["required_images"]}:
        raise ValueError("Plan required-image inventory disagrees with episodes")
    # Check every source file before writing output, for recoverable missing downloads.
    source_paths = {}
    for iid in sorted(required_ids):
        if iid not in source_images:
            raise ValueError("Unknown COCO image ID")
        relative = source_images[iid]["file_name"]
        path = (images_dir / relative).resolve()
        if not path.is_relative_to(images_dir) or not path.is_file():
            raise FileNotFoundError(f"Missing/unsafe COCO source: {relative}")
        source_paths[iid] = path
    for ep in plan["episodes"]:
        cid, split = ep["category_id"], ep["split"]
        if assignments.get(cid) != split:
            raise ValueError("Episode category/split disagrees with plan")
        if len(ep["positive_query_image_ids"]) != plan["settings"]["positive_queries"] or len(ep["negative_query_image_ids"]) != plan["settings"]["negative_queries"]:
            raise ValueError("Episode query counts disagree with plan settings")
        for present, key in ((True, "positive_query_image_ids"), (False, "negative_query_image_ids")):
            if any((cid in labels[iid]) != present for iid in ep[key]):
                raise ValueError("Query category presence disagrees with COCO annotations")
        for iid in [ep["support_image_id"], *ep["positive_query_image_ids"], *ep["negative_query_image_ids"]]:
            if any(name in assignments and assignments[name] != split for name in labels[iid]):
                raise ValueError("Cross-split category exposure")
        a = _support_annotation(ep["support_image_id"], cid, category_names, annotations, source_images,
                                plan["settings"]["min_support_area"], plan["settings"]["min_support_edge"])
        if a is None or a["id"] != ep["support_annotation_id"] or list(a["bbox"]) != ep["support_bbox_xywh"]:
            raise ValueError("Support annotation does not match fixed largest eligible target box")
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "images").mkdir()
    registry = {}; records = []
    for iid in sorted(required_ids):
        source = source_paths[iid]; rgb = load_rgb(source); original_shape = rgb.shape
        meta = source_images[iid]
        if rgb.shape[1] != meta["width"] or rgb.shape[0] != meta["height"]:
            raise ValueError(f"Decoded COCO dimensions disagree for image {iid}")
        source_pixels = pixel_sha256(rgb); source_categories = sorted(labels[iid])
        crop = None
        if iid in supports:
            ep = supports[iid]; category = ep["category_id"]; x, y, w, h = ep["support_bbox_xywh"]
            x0, y0 = max(0, math.floor(x-support_padding*w)), max(0, math.floor(y-support_padding*h))
            x1, y1 = min(rgb.shape[1], math.ceil(x+w+support_padding*w)), min(rgb.shape[0], math.ceil(y+h+support_padding*h))
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Invalid support crop")
            crop = [x0, y0, x1, y1]; rgb = rgb[y0:y1, x0:x1]
            image_categories = [category]
        else:
            selected = sorted(name for name in source_categories if name in assignments)
            if not selected:
                raise ValueError("Query source needs a selected same-split representative category")
            category = selected[0]; image_categories = source_categories
        rgb = _resize(rgb, size)
        relative = f"images/coco_{iid:012d}_{'support' if iid in supports else 'query'}.png"
        Image.fromarray(rgb).save(out_dir / relative)
        record = {"image_id": f"coco_image_{iid}", "category_id": category,
                  "category_ids": image_categories, "source_category_ids": source_categories,
                  "path": relative, "sha256": file_sha256(out_dir / relative), "pixel_sha256": pixel_sha256(rgb),
                  "source_relative_path": meta["file_name"], "source_sha256": file_sha256(source),
                  "source_pixel_sha256": source_pixels, "source_record_id": f"coco2017:{iid}",
                  "instance_id": None, "capture_id": None, "coco_image_id": iid,
                  "original_width": int(original_shape[1]), "original_height": int(original_shape[0]),
                  "stored_width": int(rgb.shape[1]), "stored_height": int(rgb.shape[0]),
                  "support_crop_xyxy_original": crop}
        if crop is not None:
            record["support_annotation_id"] = supports[iid]["support_annotation_id"]
        records.append(record); registry[iid] = record
    pairs = []
    for ep in plan["episodes"]:
        cid, episode = ep["category_id"], ep["episode_id"]
        support = registry[ep["support_image_id"]]
        for present, key in ((True, "positive_query_image_ids"), (False, "negative_query_image_ids")):
            for index, iid in enumerate(ep[key]):
                query = registry[iid]; boxes = []
                if present:
                    sx, sy = query["stored_width"]/query["original_width"], query["stored_height"]/query["original_height"]
                    for a in annotations[iid]:
                        if category_names[a["category_id"]] != cid:
                            continue
                        x, y, w, h = a["bbox"]
                        x0, y0 = max(0., x*sx), max(0., y*sy)
                        x1, y1 = min(float(query["stored_width"]), (x+w)*sx), min(float(query["stored_height"]), (y+h)*sy)
                        if x1 > x0 and y1 > y0: boxes.append([x0, y0, x1, y1])
                pairs.append({"id": f"{episode}_{'present' if present else 'absent'}_{index:03d}",
                      "category_id": cid, "query_category_id": query["category_id"],
                      "query_category_ids": query["category_ids"], "episode_id": episode,
                      "support_id": support["image_id"], "reference": support["path"], "query": query["path"],
                      "support_image_id": support["image_id"], "query_image_id": query["image_id"],
                      "support_instance_id": None, "query_instance_id": None,
                      "support_capture_id": None, "query_capture_id": None,
                      "reference_sha256": support["sha256"], "query_sha256": query["sha256"],
                      "reference_pixel_sha256": support["pixel_sha256"], "query_pixel_sha256": query["pixel_sha256"],
                      "present": present, "split": ep["split"], "condition": "independent_photo_category_presence",
                      "evaluation_annotations": {"query_target_boxes": boxes} if present else {}})
    manifest = {"schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "input_track": "full_photo_no_query_annotation",
                "seed": plan["seed"], "categories": plan["categories"], "images": records, "pairs": pairs,
                "dataset": "COCO 2017 category-presence subset", "annotation_sha256": plan["annotation_sha256"],
                "category_taxonomy": "Original COCO category labels; selected categories disjoint across dev/test; label text is not model input.",
                "settings": {**plan["settings"], "max_image_side": size, "support_padding_fraction": support_padding},
                "selection_audit": plan["selection_audit"], "annotation_budget": plan["annotation_budget"],
                "notes": ["One source support photograph and one target box per episode; the box identifies the support subject.",
                          "Full query photos are model inputs. All COCO category annotations define category presence; query boxes are evaluation only.",
                          "Support sources are globally reserved from queries. Photos exposing any selected opposite-split category are excluded.",
                          "One selected support crop is not a fully automatic unannotated-support system.",
                          "Query photos can repeat across episodes. Report category-cluster uncertainty rather than treating all pair rows as independent."],
                "limitations": ["COCO annotations can be incomplete; target-absent labels mean absent from the supplied annotations.",
                                "Distinct source photos do not establish different physical instances or independent capture sessions; those IDs are unknown.",
                                "Selected categories may have appeared in pretrained encoders; category-disjoint evaluation is not proof of globally unseen categories.",
                                "Support subjects are selected using annotated boxes and fixed minimum area/edge requirements; this is an extra annotation resource."]}
    manifest["validation"] = validate_category_manifest(manifest, out_dir)
    target = out_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "selection_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return target

