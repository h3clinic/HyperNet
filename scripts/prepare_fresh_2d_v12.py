"""Freeze and build the v12 fresh-only procedural 2D detection cohort.

Generation requires the separately reviewed renderer hash and frozen sources.
Plan selection does not inspect model performance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREVIOUS_ROOT = ROOT.parent / "hypernet_v11"
DATA = ROOT / "data/shapes2d_fresh_v12"
FREEZE = ROOT / "experiments/fresh_data_freeze.json"
AUDIT = ROOT / "experiments/fresh_data_audit.json"
ADMISSION = ROOT / "experiments/fresh_generation_admission.json"
RENDERER_MODULE = "hypernet.mathematical_shapes2d"
SPLIT_SALT = "HyperNet-v12-2D-fresh-family-split-1:"
SEED_SALT = "v12-fresh-2d"
SPLIT_COUNTS = {"dev": 128, "test": 128}
SUPPORT_SIZE = 96
QUERY_SIZE = 256
WORKERS = 6


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def seeded_int(label):
    return int.from_bytes(sha256(label.encode()).digest()[:8], "little") % (2**63)


def family_digits(family_id):
    if not isinstance(family_id, int) or not 0 <= family_id < 4096:
        raise ValueError("family_id must be an integer in [0,4096)")
    return [(family_id // power) % 8 for power in (512, 64, 8, 1)]


def prior_exposure():
    previous_plan = read(PREVIOUS_ROOT / "data/shapes2d_v11/plan.json")
    preview = read(PREVIOUS_ROOT / "assets/grammar_preview/manifest.json")
    reserved = set(previous_plan["family_partitions"]["reserved"])
    preview_ids = {e["family_id"] for e in preview["episodes"]}
    preview_ids |= {b["family_id"] for e in preview["episodes"] for q in e["queries"] for b in q["all_boxes"]}
    preview_reserved = sorted(reserved & preview_ids)
    smoke_reserved = previous_plan["family_partitions"]["reserved"][:6]
    excluded = sorted(set(preview_reserved) | set(smoke_reserved))
    require(len(reserved)==1792 and len(preview_reserved)==23 and len(excluded)==29,"Prior exposure inventory changed")
    return previous_plan, {"prior_reserved_count":1792,"preview_reserved_family_ids":preview_reserved,
                           "smoke_reserved_family_ids":smoke_reserved,"excluded_family_ids":excluded,
                           "reasons":["Rendered v11 preview object, including distractors", "Reserved-only builder integration smoke render"],
                           "canonical_census_note":"The prior all-family canonical-mask census was provenance validation, not gradient or model-score exposure."}


def family_partition():
    previous, exposure = prior_exposure()
    eligible=set(previous["family_partitions"]["reserved"])-set(exposure["excluded_family_ids"])
    ordered=sorted(eligible,key=lambda family:sha256((SPLIT_SALT+str(family)).encode()).digest())
    require(len(ordered)==1763,"Wrong eligible reserved pool")
    return {"dev":sorted(ordered[:128]),"test":sorted(ordered[128:256]),"reserved":sorted(ordered[256:])}


def make_plan():
    partitions = family_partition()
    episodes = []
    families = []
    for split in ("dev", "test"):
        for family_id in partitions[split]:
            family_key = f"v12_{split}_f{family_id:04d}"
            object_seed = seeded_int(f"{SEED_SALT}:{family_key}:object")
            scene_seed = seeded_int(f"{SEED_SALT}:{family_key}:scene")
            distractors = sorted((f for f in partitions[split] if f != family_id),
                                 key=lambda other: sha256(f"{SEED_SALT}:{family_key}:distractor:{other}".encode()).digest())[:5]
            base = {"split": split, "family_id": family_id, "grammar_digits": family_digits(family_id),
                    "object_seed": object_seed, "scene_seed": scene_seed,
                    "distractor_family_ids": distractors}
            stem = f"{split}/f{family_id:04d}"
            family_episodes = []
            for enrolled in (0, 1):
                episode_id = f"{family_key}_e{enrolled}"
                queries = []
                for present in (True, False):
                    candidate = enrolled if present else 1 - enrolled
                    label = "positive" if present else "negative"
                    queries.append({"id": f"{episode_id}_{label}", "present": present,
                                    "candidate_variant": candidate,
                                    "rgb_path": f"{stem}/query_v{candidate}.rgb.png",
                                    "target_path": f"{stem}/e{enrolled}_{label}.targets.npz"})
                episodes.append({**base, "episode_id": episode_id, "enrolled_variant": enrolled,
                                 "support": {"rgb_path": f"{stem}/support_v{enrolled}.rgb.png",
                                             "mask_path": f"{stem}/support_v{enrolled}.mask.png"},
                                 "queries": queries})
                family_episodes.append(episode_id)
            families.append({**base, "episode_ids": family_episodes,
                             "metadata_path": f"{stem}/family.metadata.json"})
    require(len(episodes)==512 and len(families)==256,"Fresh-only roster size mismatch")
    assert len({f["object_seed"] for f in families}) == len(families)
    assert len({f["scene_seed"] for f in families}) == len(families)
    previous,exposure=prior_exposure()
    old_ids={f["family_id"] for f in previous["families"]}
    old_object_seeds={f["object_seed"] for f in previous["families"]}
    old_scene_seeds={f["scene_seed"] for f in previous["families"]}
    require(not old_ids & {f["family_id"] for f in families},"Prior family exposure")
    require(not old_object_seeds & {f["object_seed"] for f in families},"Prior object seed exposure")
    require(not old_scene_seeds & {f["scene_seed"] for f in families},"Prior scene seed exposure")
    coverage={split:{axis:dict(sorted(Counter(family_digits(f)[i] for f in partitions[split]).items()))
                       for i,axis in enumerate(("outline","layout","stroke_profile","microprogram"))}
              for split in ("dev","test")}
    require(all(len(levels)==8 for axes in coverage.values() for levels in axes.values()),"Missing grammar factor level")
    return {
        "format": "hypernet.shapes2d.fresh_plan.v12", "support_size": SUPPORT_SIZE, "query_size": QUERY_SIZE,
        "family_universe": 4096, "family_partitions": partitions,
        "partition_salt": SPLIT_SALT, "seed_salt":SEED_SALT,"partition_counts":{k:len(v) for k,v in partitions.items()},
        "prior_exposure_exclusions":exposure,"factor_level_counts":coverage,
        "families": families, "episodes": episodes,
        "counts": {"families": 256, "episodes": 512, "support_images": 512,
                   "query_rows": 1024, "unique_query_images": 512, "unique_rgb_images": 1024,
                   "stored_files":2816,"training_rows":0,
                   "episodes_by_split": dict(Counter(e["split"] for e in episodes))},
        "enrollment_counterbalance": "Identical candidate scene pixels occur once positive and once negative under the two distinct support microvariants.",
        "distractor_partition": "Every distractor grammar family is from the same split as the enrolled family.",
        "split_rule": "Entire grammar family IDs, object seeds and decoded RGB pixels are disjoint across splits.",
        "natural_images": 0, "external_assets": 0, "pretrained_models": False,
        "development_access":"Develop only on v11 TRAIN/old DEV. Fresh DEV is calibration-only; fresh TEST is evaluation-only.",
        "render_gate":"No fresh image generation before explicit model/protocol freeze hash admission.",
        "scope": "New heldout grammar combinations from shared mathematical building blocks; not new natural semantic categories.",
    }


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_digest(array):
    array = np.ascontiguousarray(array)
    return sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def tight_bbox(mask):
    yy, xx = np.where(mask)
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1] if len(xx) else None


def validate_rgb_mask(rgb, mask, size):
    require(rgb.shape == (size, size, 3) and rgb.dtype == np.uint8, "RGB native size/dtype mismatch")
    require(mask.shape == (size, size) and mask.dtype == np.bool_, "Mask native size/dtype mismatch")
    require(0 < int(mask.sum()) < size * size, "Foreground must be nonempty and retain background")


def json_metadata(value):
    if isinstance(value, dict):
        return {str(k): json_metadata(v) for k, v in value.items()
                if not (isinstance(v, np.ndarray) and v.ndim > 1)}
    if isinstance(value, (list, tuple)):
        return [json_metadata(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_png(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        Image.fromarray(array).save(stream, format="PNG")


def build_prior_rgb_inventory():
    previous=read(PREVIOUS_ROOT/'data/shapes2d_v11/manifest.json')
    preview=read(PREVIOUS_ROOT/'assets/grammar_preview/manifest.json')
    planned={}
    for episode in previous['episodes']:
        for record in [episode['support']]+episode['queries']:
            path=(PREVIOUS_ROOT/'data/shapes2d_v11'/record['rgb_path']).resolve()
            planned[str(path)]={'cohort':'v11_main','expected_raw_sha256':record['rgb_sha256'],
                                'expected_rgb_array_sha256':record['rgb_array_sha256']}
    for episode in preview['episodes']:
        for record in [episode['support']]+episode['queries']:
            path=(PREVIOUS_ROOT/'assets/grammar_preview'/record['rgb_path']).resolve()
            planned[str(path)]={'cohort':'v11_preview'}
    require(len(planned)==9264,'Prior main/preview source count changed')
    rows=[]
    for path,info in sorted(planned.items()):
        raw=digest(path)
        with Image.open(path) as image:rgb=np.array(image.convert('RGB'))
        pixels=array_digest(rgb)
        if info['cohort']=='v11_main':
            require(raw==info['expected_raw_sha256'] and pixels==info['expected_rgb_array_sha256'],'Prior v11 source changed')
        rows.append({'path':path,'cohort':info['cohort'],'raw_sha256':raw,'rgb_array_sha256':pixels,'shape':list(rgb.shape)})
    return {'format':'hypernet.prior_2d_rgb_inventory.v12','files':rows,
            'file_count':len(rows),'unique_raw_sha256_count':len({r['raw_sha256'] for r in rows}),
            'unique_rgb_array_sha256_count':len({r['rgb_array_sha256'] for r in rows}),
            'scope':'Original v11 main data and preview RGB files; smoke programs excluded by family ID because their temporary images are not retained.'}


def plan_stage(expected_renderer_sha):
    renderer = importlib.import_module(RENDERER_MODULE)
    renderer_path = Path(renderer.__file__).resolve()
    require(digest(renderer_path) == expected_renderer_sha, "Renderer differs from reviewed hash")
    require(renderer.FAMILY_COUNT == 4096, "Renderer grammar universe mismatch")
    census_path = PREVIOUS_ROOT / "experiments/canonical_2d_grammar_audit.json"
    census = read(census_path)
    require(census["passed"] and census["renderer_sha256"] == expected_renderer_sha,
            "Final renderer canonical-grammar census must pass before generation")
    require(census["microstates_checked"] == 8192 and not census["duplicate_canonical_masks"] and
            not census["duplicate_up_to_right_angle_rotation"], "Grammar distinctness census mismatch")
    plan = make_plan()
    plan["renderer_module"] = RENDERER_MODULE
    plan["grammar_axes"] = ["outline", "layout", "stroke_profile", "microprogram"]
    for family in plan["families"]:
        require(list(renderer.decode_family(family["family_id"])) == family["grammar_digits"], "Grammar digit interpretation mismatch")
    inventory=build_prior_rgb_inventory()
    inventory_path=DATA/'prior_rgb_inventory.json'
    write_new(inventory_path,inventory)
    plan['prior_rgb_inventory']={'path':str(inventory_path),'sha256':digest(inventory_path),'files':inventory['file_count']}
    write_new(DATA / "plan.json", plan)
    paths = [renderer_path, Path(__file__).resolve(), ROOT / "data/SHAPES2D_FRESH_CONTRACT.json"]
    prior_paths=[PREVIOUS_ROOT/'data/shapes2d_v11/plan.json',PREVIOUS_ROOT/'data/shapes2d_v11/manifest.json',
                 PREVIOUS_ROOT/'assets/grammar_preview/manifest.json',PREVIOUS_ROOT/'tests/test_2d_dataset.py',
                 PREVIOUS_ROOT/'experiments/shapes2d_evidence_bundle.json',inventory_path]
    freeze = {"format": "hypernet.shapes2d.fresh_data_freeze.v12", "created_utc": datetime.now(timezone.utc).isoformat(),
              "plan": {"path": str(DATA / "plan.json"), "sha256": digest(DATA / "plan.json")},
              "behavior_files": {str(p): digest(p) for p in paths}, "workers": WORKERS,
              "prior_evidence_files":{str(p):digest(p) for p in prior_paths},
              "renderer_validation": {"path":str(census_path),"sha256":digest(census_path)},
              "generation_started_before_freeze": False, "model_scores_used_for_selection": False,
              "natural_images": 0, "external_assets": 0, "external_pretrained_weights": False,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": pillow_version}}
    write_new(FREEZE, freeze)
    print(json.dumps({"freeze": str(FREEZE), "counts": plan["counts"]}), flush=True)


def check_freeze():
    freeze = read(FREEZE)
    require(digest(freeze["plan"]["path"]) == freeze["plan"]["sha256"], "Frozen generation plan changed")
    for path, expected in freeze["behavior_files"].items():
        require(digest(path) == expected, "Frozen generation source changed: " + path)
    for path,expected in freeze['prior_evidence_files'].items():
        require(digest(path)==expected,'Frozen prior-data inventory changed: '+path)
    require(digest(freeze["renderer_validation"]["path"]) == freeze["renderer_validation"]["sha256"],
            "Frozen renderer validation changed")
    return freeze


def admit_generation(protocol_path,expected_protocol_sha):
    require(protocol_path is not None and expected_protocol_sha,'Explicit model/protocol freeze path and SHA required')
    protocol_path=Path(protocol_path).resolve()
    require(protocol_path.is_file() and digest(protocol_path)==expected_protocol_sha,'Model/protocol freeze missing or changed')
    protocol=read(protocol_path)
    require(protocol.get('model_selection_complete') is True,'Model selection must be complete before fresh generation')
    require(protocol.get('fresh_test_generated_before_freeze') is False,'Protocol must declare no prior fresh TEST generation')
    require(protocol.get('fresh_plan_sha256')==digest(DATA/'plan.json'),'Protocol must bind this fresh plan')
    source_bindings=protocol.get('behavior_files',{})
    require(bool(source_bindings),'Protocol must bind model/scorer behavior sources')
    for path,expected in source_bindings.items():
        resolved=Path(path) if Path(path).is_absolute() else ROOT/path
        require(digest(resolved)==expected,'Model/scorer source changed before generation')
    admission={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_path':str(protocol_path),
               'protocol_sha256':expected_protocol_sha,'data_freeze_sha256':digest(FREEZE),
               'fresh_plan_sha256':digest(DATA/'plan.json'),'no_fresh_images_before_admission':not any(DATA.rglob('*.png')),
               'model_selection_complete_before_generation':True}
    require(admission['no_fresh_images_before_admission'],'Fresh images already exist before generation admission')
    write_new(ADMISSION,admission)
    return admission


def check_admission():
    admission=read(ADMISSION)
    require(digest(admission['protocol_path'])==admission['protocol_sha256'],'Frozen model/protocol changed')
    require(digest(FREEZE)==admission['data_freeze_sha256'],'Frozen data protocol changed')
    protocol=read(admission['protocol_path'])
    for path,expected in protocol['behavior_files'].items():
        resolved=Path(path) if Path(path).is_absolute() else ROOT/path
        require(digest(resolved)==expected,'Frozen model/scorer behavior changed during data generation')
    return admission


def normalize_boxes(query, family_id):
    records = query["object_records"]
    boxes = query["object_boxes"]
    require(len(records) == len(boxes) == 6, "Expected one candidate and five distractors")
    output = []
    for record, box in zip(records, boxes):
        fid = int(record["family_id"])
        variant = record.get("micro_variant", record.get("microvariant", record.get("variant")))
        require(variant in (0, 1), "Object microvariant missing or invalid")
        bbox = record.get("bbox_xyxy", box)
        if isinstance(bbox, dict):
            bbox = bbox["bbox_xyxy"]
        output.append({"bbox_xyxy": [int(v) for v in bbox], "role": "candidate" if fid == family_id else "distractor",
                       "family_id": fid, "micro_variant": int(variant)})
    require(sum(b["role"] == "candidate" for b in output) == 1, "Exactly one candidate required")
    return output


def render_family(job):
    family, planned_episodes = job
    renderer = importlib.import_module(RENDERER_MODULE)
    start = time.perf_counter()
    rendered = [renderer.render_episode(family["family_id"], family["object_seed"], enrolled,
                                        family["scene_seed"], family["distractor_family_ids"],
                                        support_size=SUPPORT_SIZE, query_size=QUERY_SIZE) for enrolled in (0, 1)]
    for enrolled,frame in enumerate(rendered):
        require(frame["family_spec"]["family_id"]==family["family_id"] and
                frame["family_spec"]["grammar_digits"]==family["grammar_digits"], "Renderer family identity mismatch")
        require(frame["pair_statistics"]["query_candidate_is_new_macro_instance"] is True,
                "Reference and candidate must use independently jittered macro instances")
    candidate_record=next(r for r in rendered[0]["queries"]["0"]["object_records"] if r["family_id"]==family["family_id"])
    require(candidate_record["instance_seed"] != family["object_seed"], "Reference/candidate instance seed reused")
    support_parameters=rendered[0]["parameters"]["support_parameters"]
    require(candidate_record["parameters"] != support_parameters, "Reference/candidate macro geometry identical")
    for parameters in (support_parameters,candidate_record["parameters"]):
        require(all(.97<=v<=1.03 for v in parameters["axis_jitter"]) and .97<=parameters["connector_gain"]<=1.03,
                "Instance variation exceeds frozen small-jitter range")
    for candidate in (0, 1):
        first, second = (r["queries"][str(candidate)] for r in rendered)
        for key in ("query_rgb", "union_mask", "candidate_mask"):
            require(np.array_equal(first[key], second[key]), "Enrollment leaked into query pixels/geometry")
        require(first["matches_reference"] is (candidate == 0) and second["matches_reference"] is (candidate == 1),
                "Renderer reference-match labels are not counterbalanced")
    require(not np.array_equal(rendered[0]["support_rgb"], rendered[1]["support_rgb"]), "Support microvariants must differ")
    q0, q1 = (rendered[0]["queries"][str(v)] for v in (0, 1))
    changed = np.any(q0["query_rgb"] != q1["query_rgb"], axis=-1)
    candidate_union = q0["candidate_mask"] | q1["candidate_mask"]
    require(changed.any() and not changed[~candidate_union].any(), "Paired scene change must be confined to candidate")
    require(np.array_equal(q0["union_mask"] & ~candidate_union, q1["union_mask"] & ~candidate_union),
            "Distractor masks changed across variants")
    stats0,stats1 = q0["micro_statistics"],q1["micro_statistics"]
    require(stats0["primitive_count"] == stats1["primitive_count"] == 2, "Microprimitive budget mismatch")
    require(abs(stats0["continuous_removed_area"] - stats1["continuous_removed_area"]) < 1e-9,
            "Microvariants must have identical continuous roughness/removed-area budget")
    require(stats0["minimum_nominal_feature_width_pixels"] > 0 and stats1["minimum_nominal_feature_width_pixels"] > 0,
            "Microfeatures must have positive extent")
    results = []
    for enrolled, (episode, frame) in enumerate(zip(planned_episodes, rendered)):
        support = dict(episode["support"])
        validate_rgb_mask(frame["support_rgb"], frame["support_mask"], SUPPORT_SIZE)
        support_box = tight_bbox(frame["support_mask"])
        require(list(frame["support_bbox_xyxy"]) == support_box, "Support box must tightly enclose foreground")
        save_png(DATA / support["rgb_path"], frame["support_rgb"])
        save_png(DATA / support["mask_path"], frame["support_mask"].astype(np.uint8) * 255)
        support.update(bbox_xyxy=support_box, rgb_sha256=digest(DATA / support["rgb_path"]),
                       mask_sha256=digest(DATA / support["mask_path"]), rgb_array_sha256=array_digest(frame["support_rgb"]),
                       mask_array_sha256=array_digest(frame["support_mask"]))
        queries = []
        for planned in episode["queries"]:
            row = dict(planned)
            query = frame["queries"][str(row["candidate_variant"]) ]
            validate_rgb_mask(query["query_rgb"], query["union_mask"], QUERY_SIZE)
            validate_rgb_mask(query["query_rgb"], query["candidate_mask"], QUERY_SIZE)
            candidate_box = tight_bbox(query["candidate_mask"])
            require(list(query["candidate_bbox_xyxy"]) == candidate_box, "Candidate box mismatch")
            expected_target = query["candidate_mask"] if row["present"] else np.zeros_like(query["candidate_mask"])
            require(np.array_equal(query["target_mask"], expected_target), "Target mask violates enrolled identity")
            require(query["target_bbox_xyxy"] == (candidate_box if row["present"] else None), "Target box violates enrolled identity")
            boxes = normalize_boxes(query, family["family_id"])
            require({b["family_id"] for b in boxes if b["role"] == "distractor"} == set(family["distractor_family_ids"]),
                    "Renderer changed planned distractor families")
            path = DATA / row["rgb_path"]
            if enrolled == 0:
                save_png(path, query["query_rgb"])
            else:
                with Image.open(path) as saved:
                    require(np.array_equal(np.array(saved), query["query_rgb"]), "Counterbalanced shared RGB mismatch")
            targets = {key: query[key] for key in ("union_mask", "target_mask", "candidate_mask")}
            with (DATA / row["target_path"]).open("xb") as stream:
                np.savez_compressed(stream, **targets)
            row.update(target_bbox_xyxy=candidate_box if row["present"] else None, candidate_bbox_xyxy=candidate_box,
                       all_boxes=boxes, rgb_sha256=digest(path), rgb_array_sha256=array_digest(query["query_rgb"]),
                       target_sha256=digest(DATA / row["target_path"]),
                       target_array_sha256={k: array_digest(v) for k,v in targets.items()})
            queries.append(row)
        results.append({**episode, "support": support, "queries": queries})
    metadata_path = DATA / family["metadata_path"]
    write_new(metadata_path, {"family": family, "enrollments": [json_metadata(r) for r in rendered]})
    report = {**family, "metadata_sha256": digest(metadata_path), "paired_changed_pixels": int(changed.sum()),
              "canonical_program_id":rendered[0]["family_spec"]["canonical_program_id"],
              "family_name":rendered[0]["family_spec"]["family_name"],
              "candidate_instance_seed":candidate_record["instance_seed"],
              "candidate_instance_key":candidate_record["instance_key"],
              "reference_query_macro_parameters_differ":True,
              "candidate_pixels": [int(q["candidate_mask"].sum()) for q in (q0,q1)],
              "micro_statistics": [json_metadata(stats0),json_metadata(stats1)],
              "continuous_removed_area_equal": True,
              "counterbalance_exact": True, "pair_changes_candidate_only": True,
              "elapsed_seconds": time.perf_counter() - start}
    return report, results


def render_stage(protocol_path,expected_protocol_sha):
    freeze = check_freeze()
    plan = read(freeze["plan"]["path"])
    require(not (DATA / "manifest.json").exists(), "Preserve completed manifest")
    admit_generation(protocol_path,expected_protocol_sha)
    episode_map = {e["episode_id"]: e for e in plan["episodes"]}
    jobs = [(f,[episode_map[e] for e in f["episode_ids"]]) for f in plan["families"]]
    started = time.perf_counter()
    families = {}; episodes = {}
    with ProcessPoolExecutor(max_workers=freeze["workers"], initializer=check_freeze) as pool:
        futures = [pool.submit(render_family, job) for job in jobs]
        for future in as_completed(futures):
            family, rows = future.result()
            families[family["family_id"]] = family
            episodes.update((e["episode_id"], e) for e in rows)
            if len(families) % 64 == 0:
                print(f"Rendered {len(families)}/{len(jobs)} families ({4*len(families)} unique RGB images)", flush=True)
    check_freeze();check_admission()
    require(len(families)==plan['counts']['families'] and len(episodes)==plan['counts']['episodes'], "Incomplete dataset render")
    manifest = {**plan, "format": "hypernet.shapes2d.fresh_dataset.v12", "completed": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "families": [families[f["family_id"]] for f in plan["families"]],
                "episodes": [episodes[e["episode_id"]] for e in plan["episodes"]],
                "freeze_sha256": digest(FREEZE),"generation_admission_sha256":digest(ADMISSION), "elapsed_seconds": time.perf_counter() - started}
    write_new(DATA / "manifest.json", manifest)
    print(json.dumps({"completed": True, "manifest": str(DATA / "manifest.json"), "seconds": manifest["elapsed_seconds"]}), flush=True)


def audit_stage():
    freeze = check_freeze(); plan = read(freeze["plan"]["path"]); manifest = read(DATA / "manifest.json")
    admission=check_admission()
    require(manifest["completed"] is True and manifest["freeze_sha256"] == digest(FREEZE), "Incomplete/unbound dataset")
    require(manifest['generation_admission_sha256']==digest(ADMISSION),'Generation admission binding changed')
    require([e["episode_id"] for e in manifest["episodes"]] == [e["episode_id"] for e in plan["episodes"]], "Episode roster changed")
    paths = set(); pixel_splits = defaultdict(set); counters = Counter(); by_family = defaultdict(list);fresh_raw_hashes=set()
    for episode, planned in zip(manifest["episodes"], plan["episodes"]):
        for key in ("episode_id","split","family_id","grammar_digits","object_seed","scene_seed","distractor_family_ids","enrolled_variant"):
            require(episode[key] == planned[key], "Planned episode field changed: " + key)
        support = episode["support"]
        for path_key, digest_key in (("rgb_path","rgb_sha256"),("mask_path","mask_sha256")):
            path = (DATA / support[path_key]).resolve()
            require(path.is_relative_to(DATA.resolve()) and digest(path) == support[digest_key], "Support path/hash mismatch")
            paths.add(path)
            if path_key=='rgb_path':fresh_raw_hashes.add(support[digest_key])
        with Image.open(DATA / support["rgb_path"]) as image: rgb = np.array(image)
        with Image.open(DATA / support["mask_path"]) as image: mask = np.array(image) > 0
        validate_rgb_mask(rgb, mask, SUPPORT_SIZE)
        require(array_digest(rgb) == support["rgb_array_sha256"] and array_digest(mask) == support["mask_array_sha256"], "Decoded support mismatch")
        require(tight_bbox(mask) == support["bbox_xyxy"], "Support bbox mismatch")
        pixel_splits[array_digest(rgb)].add(episode["split"])
        for query, query_plan in zip(episode["queries"], planned["queries"]):
            require(all(query[k] == v for k,v in query_plan.items()), "Planned query field changed")
            for path_key, digest_key in (("rgb_path","rgb_sha256"),("target_path","target_sha256")):
                path = (DATA / query[path_key]).resolve()
                require(path.is_relative_to(DATA.resolve()) and digest(path) == query[digest_key], "Query path/hash mismatch")
                paths.add(path)
                if path_key=='rgb_path':fresh_raw_hashes.add(query[digest_key])
            with Image.open(DATA / query["rgb_path"]) as image: rgb = np.array(image)
            with np.load(DATA / query["target_path"], allow_pickle=False) as targets:
                require(set(targets.files) == {"union_mask","candidate_mask","target_mask"}, "Unexpected target fields")
                arrays = {k: targets[k] for k in targets.files}
            for key, array in arrays.items():
                require(array.dtype == np.bool_ and array.shape == (QUERY_SIZE,QUERY_SIZE), "Target dtype/shape mismatch")
                require(array_digest(array) == query["target_array_sha256"][key], "Decoded target hash mismatch")
            validate_rgb_mask(rgb, arrays["union_mask"], QUERY_SIZE)
            require(not (arrays["candidate_mask"] & ~arrays["union_mask"]).any(), "Candidate lies outside union mask")
            expected = arrays["candidate_mask"] if query["present"] else np.zeros_like(arrays["candidate_mask"])
            require(np.array_equal(arrays["target_mask"], expected), "Target mask violates counterbalance")
            require(query["target_bbox_xyxy"] == tight_bbox(expected), "Target bbox mismatch")
            require(query["candidate_bbox_xyxy"] == tight_bbox(arrays["candidate_mask"]), "Candidate bbox mismatch")
            require(array_digest(rgb) == query["rgb_array_sha256"], "Decoded query RGB mismatch")
            pixel_splits[array_digest(rgb)].add(episode["split"])
            allowed = set(plan["family_partitions"][episode["split"]])
            require(all(box["family_id"] in allowed for box in query["all_boxes"]), "Distractor cross-split exposure")
            require(len(query["all_boxes"]) == 6 and sum(b["role"] == "candidate" for b in query["all_boxes"]) == 1, "Wrong object count")
            counters[episode["split"] + "_queries"] += 1
        by_family[episode["family_id"]].append(episode)
    require(not any(len(splits) > 1 for splits in pixel_splits.values()), "Decoded RGB leakage across family splits")
    pair_checks = 0
    for family_id, episodes in by_family.items():
        first, second = episodes
        require([e["enrolled_variant"] for e in episodes] == [0,1], "Enrollment roles unbalanced")
        for i in (0,1):
            a,b = first["queries"][i],second["queries"][1-i]
            require(a["rgb_path"] == b["rgb_path"] and a["rgb_array_sha256"] == b["rgb_array_sha256"] and a["present"] != b["present"], "Counterbalanced shared scene mismatch")
            pair_checks += 1
        with Image.open(DATA/first["queries"][0]["rgb_path"]) as image: rgb0=np.array(image)
        with Image.open(DATA/first["queries"][1]["rgb_path"]) as image: rgb1=np.array(image)
        with np.load(DATA/first["queries"][0]["target_path"]) as target: mask0=target["candidate_mask"]
        with np.load(DATA/first["queries"][1]["target_path"]) as target: mask1=target["candidate_mask"]
        changes=np.any(rgb0 != rgb1,axis=-1)
        require(changes.any() and not changes[~(mask0|mask1)].any(), "Paired scene changed outside candidate")
    for family in manifest["families"]:
        path=DATA/family["metadata_path"]
        require(digest(path)==family["metadata_sha256"], "Family metadata hash changed")
        stats0,stats1=family["micro_statistics"]
        require(stats0["primitive_count"]==stats1["primitive_count"]==2 and
                abs(stats0["continuous_removed_area"]-stats1["continuous_removed_area"])<1e-9,
                "Microvariant equal-budget control failed")
        paths.add(path.resolve())
    require(len(paths)==plan['counts']['stored_files'], "Unexpected stored file count")
    require(len({f["canonical_program_id"] for f in manifest["families"]})==plan['counts']['families'],"Canonical family program collision")
    require(len({f["candidate_instance_key"] for f in manifest["families"]})==plan['counts']['families'],"Query candidate instance collision")
    prior=read(plan['prior_rgb_inventory']['path'])
    require(digest(plan['prior_rgb_inventory']['path'])==plan['prior_rgb_inventory']['sha256'],'Prior inventory binding mismatch')
    old_raw={row['raw_sha256'] for row in prior['files']};old_pixels={row['rgb_array_sha256'] for row in prior['files']}
    require(not old_raw & fresh_raw_hashes,'Fresh images duplicate a prior source file')
    require(not old_pixels & set(pixel_splits),'Fresh images duplicate prior decoded RGB pixels')
    for row in prior['files']:
        require(digest(row['path'])==row['raw_sha256'],'Prior source file changed since inventory')
    require(not any(e['split']=='train' for e in manifest['episodes']),'Fresh cohort may not contain training rows')
    check_freeze()
    audit={"valid":True,"created_utc":datetime.now(timezone.utc).isoformat(),"manifest_sha256":digest(DATA/"manifest.json"),
           "freeze_sha256":digest(FREEZE),"auditor_sha256":digest(__file__),"files_verified":len(paths),
           "episodes_verified":len(manifest["episodes"]),"query_rows_verified":sum(counters.values()),
           "unique_rgb_arrays":len(pixel_splits),"same_scene_opposite_enrollment_checks":pair_checks,
           "counts":dict(counters),"whole_grammar_family_split_disjoint":True,"distractors_within_split_only":True,
           "prior_rgb_files_excluded":len(prior['files']),"prior_raw_file_overlap":0,"prior_decoded_rgb_overlap":0,
           "prior_main_and_preview_family_exposure_excluded":True,"training_rows":0,
           "model_protocol_frozen_before_generation":True,"generation_admission_sha256":digest(ADMISSION),
           "model_protocol_sha256":admission['protocol_sha256'],
           "cross_split_pixel_overlap":0,"paired_scene_changes_confined_to_candidate":True,
           "continuous_microfeature_removed_area_equal":True,
           "independently_jittered_reference_query_instances":plan['counts']['families'],
           "rasterized_candidate_area_difference_max":max(abs(f["candidate_pixels"][0]-f["candidate_pixels"][1]) for f in manifest["families"]),
           "minimum_nominal_feature_width_pixels":min(s["minimum_nominal_feature_width_pixels"] for f in manifest["families"] for s in f["micro_statistics"]),
           "query_annotations_are_evaluation_targets_not_model_inputs":True,"natural_images":0,"external_assets":0}
    write_new(AUDIT,audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("plan","render","audit"),required=True)
    parser.add_argument("--expected-renderer-sha")
    parser.add_argument('--protocol-freeze',type=Path)
    parser.add_argument('--expected-protocol-sha')
    args=parser.parse_args()
    if args.stage=="plan":
        require(bool(args.expected_renderer_sha),"Reviewed renderer SHA is required before freeze")
        plan_stage(args.expected_renderer_sha)
    elif args.stage=="render": render_stage(args.protocol_freeze,args.expected_protocol_sha)
    else: audit_stage()
