import copy
import json

import numpy as np
import pytest
from PIL import Image

from hypernet.category_coco import build_coco_category_dataset
from hypernet.category_fresh_v09 import plan_fresh_coco_categories_v09, audit_fresh_category_manifest_v09
from hypernet.data import file_sha256, load_rgb, pixel_sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def cohort(tmp_path):
    images_dir = tmp_path / "sources"
    images_dir.mkdir()
    rng = np.random.default_rng(7)
    categories = [{"id": i + 1, "name": f"category_{i}"} for i in range(6)] + [{"id": 99, "name": "person"}]
    images, annotations, old_images = [], [], []
    for c in range(6):
        for n in range(8):
            iid = c * 100 + n + 1
            name = f"{iid}.png"
            dest = images_dir / name
            Image.fromarray(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)).save(dest)
            images.append({"id": iid, "file_name": name, "width": 96, "height": 80,
                           "coco_url": "http://images.cocodataset.org/val2017/" + name, "license": 1})
            for cid in [c + 1, 99]:
                annotations.append({"id": len(annotations) + 1, "image_id": iid, "category_id": cid,
                                    "bbox": [10, 10, 50, 45], "iscrowd": 0, "area": 2250})
            if n == 0:
                old_images.append({"coco_image_id": iid, "source_relative_path": name,
                                   "source_sha256": file_sha256(dest), "source_pixel_sha256": pixel_sha256(load_rgb(dest))})
    raw = {"images": images, "annotations": annotations, "categories": categories,
           "licenses": [{"id": 1, "name": "fixture", "url": "https://example.invalid/license"}]}
    original = write(tmp_path / "original.json", raw)
    prior = {"annotation_sha256": file_sha256(original), "images": old_images,
             "categories": [{"category_id": f"category_{c}", "split": "dev" if c < 3 else "test"} for c in range(6)]}
    prior_path = write(tmp_path / "prior/manifest.json", prior)
    kwargs = {"category_counts": (6, 4), "episodes_per_category": 2, "positive_queries": 2,
              "negative_queries": 2, "min_support_area": 100, "min_support_edge": 16}
    return original, prior_path, images_dir, kwargs


def make_plan(cohort, tmp_path):
    original, prior, _, kwargs = cohort
    filtered, plan_path = tmp_path / "filtered.json", tmp_path / "plan.json"
    plan = plan_fresh_coco_categories_v09(original, [prior], filtered, plan_path, **kwargs)
    return plan, filtered, plan_path


def test_reuses_types_preserves_splits_and_all_annotations(cohort, tmp_path):
    original, prior_path, _, _ = cohort
    prior = json.loads(prior_path.read_text())
    plan, filtered, _ = make_plan(cohort, tmp_path)
    assert sorted(plan["categories"], key=lambda c: c["category_id"]) == prior["categories"]
    old_ids = {r["coco_image_id"] for r in prior["images"]}
    assert not old_ids & {r["id"] for r in plan["required_images"]}
    raw = json.loads(original.read_text())
    retained = json.loads(filtered.read_text())
    assert retained["categories"] == raw["categories"]
    assert retained["licenses"] == raw["licenses"]
    assert retained["annotations"] == [r for r in raw["annotations"] if r["image_id"] not in old_ids]
    assert plan["freshness"]["scores_used_for_selection"] is False
    assert len(plan["freshness"]["selected_target_category_overlap"]) == 6
    supports = {e["support_image_id"] for e in plan["episodes"]}
    queries = {i for e in plan["episodes"] for k in ("positive_query_image_ids", "negative_query_image_ids") for i in e[k]}
    assert len(supports) == 12 and not supports & queries


def test_predeclared_balanced_fallback_uses_availability_only(cohort, tmp_path):
    original, prior, _, kwargs = cohort
    raw = json.loads(original.read_text())
    keep = {r["id"] for r in raw["images"] if r["id"] <= 501}
    raw["images"] = [r for r in raw["images"] if r["id"] in keep]
    raw["annotations"] = [r for r in raw["annotations"] if r["image_id"] in keep]
    write(original, raw)
    old = json.loads(prior.read_text())
    old["annotation_sha256"] = file_sha256(original)
    write(prior, old)
    plan, _, _ = make_plan(cohort, tmp_path)
    assert [(a["categories"], a["feasible"]) for a in plan["freshness"]["cohort_size_attempts"]] == [(6, False), (4, True)]
    assert "category_5" not in {c["category_id"] for c in plan["categories"]}
    assert sum(c["split"] == "dev" for c in plan["categories"]) == 2
    previous = {c["category_id"]: c["split"] for c in old["categories"]}
    assert all(previous[c["category_id"]] == c["split"] for c in plan["categories"])


def test_selected_opposite_split_annotations_are_excluded(cohort, tmp_path):
    original, prior, _, _ = cohort
    raw = json.loads(original.read_text())
    raw["annotations"].append({"id": 9999, "image_id": 2, "category_id": 4, "bbox": [10, 10, 50, 45], "iscrowd": 0})
    write(original, raw)
    old = json.loads(prior.read_text())
    old["annotation_sha256"] = file_sha256(original)
    write(prior, old)
    plan, _, _ = make_plan(cohort, tmp_path)
    assert 2 not in {r["id"] for r in plan["required_images"]}


def test_import_and_freshness_audit_verify_actual_source_bytes(cohort, tmp_path):
    _, prior, images_dir, _ = cohort
    plan, filtered, _ = make_plan(cohort, tmp_path)
    manifest = build_coco_category_dataset(filtered, images_dir, tmp_path / "dataset", plan=plan, size=96)
    audit = audit_fresh_category_manifest_v09(manifest, [prior], source_images_dir=images_dir)
    assert audit["valid"] and not any(audit["overlaps"].values())
    assert len(audit["reused_target_categories"]) == 6
    assert audit["original_source_files_verified"] == audit["n_source_photos"] + 6
    original = images_dir / "1.png"
    Image.fromarray(np.zeros((80, 96, 3), dtype=np.uint8)).save(original)
    with pytest.raises(ValueError, match="Original source hash mismatch"):
        audit_fresh_category_manifest_v09(manifest, [prior], source_images_dir=images_dir)


@pytest.mark.parametrize("field", ["coco_image_id", "source_sha256", "source_pixel_sha256"])
def test_post_download_overlap_checks_all_source_identities(cohort, tmp_path, field):
    _, prior, images_dir, _ = cohort
    plan, filtered, _ = make_plan(cohort, tmp_path)
    manifest = build_coco_category_dataset(filtered, images_dir, tmp_path / "dataset", plan=plan, size=96)
    fresh = json.loads(manifest.read_text())
    old = json.loads(prior.read_text())
    old["images"][0][field] = fresh["images"][0][field]
    write(prior, old)
    with pytest.raises(ValueError, match="Fresh-source overlap"):
        audit_fresh_category_manifest_v09(manifest, [prior])


def test_requires_prior_hashes_and_does_not_overwrite(cohort, tmp_path):
    original, prior, _, kwargs = cohort
    old = json.loads(prior.read_text())
    old["images"][0].pop("source_sha256")
    write(prior, old)
    with pytest.raises(ValueError, match="prior source_sha256"):
        make_plan(cohort, tmp_path)
    with pytest.raises(ValueError, match="cannot replace"):
        plan_fresh_coco_categories_v09(original, [prior], original, tmp_path / "plan.json", **kwargs)


def test_nonpositive_quotas_rejected_before_writing(cohort, tmp_path):
    original, prior, _, kwargs = cohort
    kwargs["positive_queries"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        plan_fresh_coco_categories_v09(original, [prior], tmp_path / "f.json", tmp_path / "p.json", **kwargs)
    assert not (tmp_path / "f.json").exists()


def test_five_prior_cohorts_are_all_excluded(cohort, tmp_path):
    original, prior_path, _, kwargs = cohort
    previous = json.loads(prior_path.read_text())
    paths = []
    expected_ids = {r['coco_image_id'] for r in previous['images']}
    for index in range(5):
        part = copy.deepcopy(previous)
        part['images'] = previous['images'][index::5]
        paths.append(write(tmp_path / f'previous_{index}' / 'manifest.json', part))
    plan = plan_fresh_coco_categories_v09(original, paths, tmp_path/'filtered_five.json',
        tmp_path/'plan_five.json', preferred_manifest_path=paths[-1], **kwargs)
    assert set(plan['freshness']['excluded_prior_source_ids']) == expected_ids
    assert len(plan['freshness']['prior_manifests']) == 5
    assert not expected_ids & {r['id'] for r in plan['required_images']}
    assert plan['freshness']['scores_used_for_selection'] is False


def test_preferred_split_must_belong_to_excluded_cohorts(cohort, tmp_path):
    original, prior, _, kwargs = cohort
    with pytest.raises(ValueError, match='among the excluded'):
        plan_fresh_coco_categories_v09(original, [prior], tmp_path/'f.json', tmp_path/'p.json',
            preferred_manifest_path=tmp_path/'not_excluded.json', **kwargs)
    assert not (tmp_path/'f.json').exists()


def test_selection_is_seeded_and_independent_of_output_paths(cohort,tmp_path):
    original,prior,_,kwargs=cohort
    first=plan_fresh_coco_categories_v09(original,[prior],tmp_path/'a/f.json',tmp_path/'a/p.json',**kwargs)
    second=plan_fresh_coco_categories_v09(original,[prior],tmp_path/'b/f.json',tmp_path/'b/p.json',**kwargs)
    for key in ('categories','episodes','required_images','selection_audit'):
        assert first[key]==second[key]
    assert first['selection_audit']['candidate_category_subsets']==1


def test_duplicate_repaired_cohort_does_not_double_count_sources(cohort,tmp_path):
    original,prior,_,kwargs=cohort
    previous=json.loads(prior.read_text())
    repaired=write(tmp_path/'repaired/manifest.json',previous)
    plan=plan_fresh_coco_categories_v09(original,[prior,repaired],tmp_path/'f.json',tmp_path/'p.json',**kwargs)
    assert len(plan['freshness']['excluded_prior_source_ids'])==6
    assert len(plan['freshness']['prior_source_sha256'])==6
    assert len(plan['freshness']['prior_source_pixel_sha256'])==6
