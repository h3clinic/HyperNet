from collections import Counter
import importlib.util
from pathlib import Path

import pytest
import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts/prepare_2d_data.py"
spec = importlib.util.spec_from_file_location("shape2d_builder", path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture(scope="module")
def plan():
    return builder.make_plan()


def test_entire_grammar_family_split_exclusion_and_exact_budget(plan):
    parts = plan["family_partitions"]
    assert parts == builder.family_partition()
    assert {k:len(v) for k,v in parts.items()} == {"train":2048,"dev":128,"test":128,"reserved":1792}
    assert len(set(f for values in parts.values() for f in values)) == 4096
    assert plan["counts"]["episodes_by_split"] == {"train":4096,"dev":256,"test":256}
    for split in ("train", "dev", "test"):
        for axis in range(4):
            assert set(builder.family_digits(f)[axis] for f in parts[split]) == set(range(8))


def test_distractors_cannot_cross_split_or_equal_reference_family(plan):
    for family in plan["families"]:
        allowed = set(plan["family_partitions"][family["split"]])
        assert len(family["distractor_family_ids"]) == 5
        assert family["family_id"] not in family["distractor_family_ids"]
        assert set(family["distractor_family_ids"]) <= allowed
    assert len({f["object_seed"] for f in plan["families"]}) == 2304


def test_each_scene_is_label_counterbalanced_only_by_support(plan):
    for e0, e1 in zip(plan["episodes"][::2], plan["episodes"][1::2]):
        assert e0["family_id"] == e1["family_id"]
        assert e0["enrolled_variant"] == 0 and e1["enrolled_variant"] == 1
        assert e0["object_seed"] == e1["object_seed"] and e0["scene_seed"] == e1["scene_seed"]
        assert e0["support"]["rgb_path"] != e1["support"]["rgb_path"]
        assert e0["queries"][0]["rgb_path"] == e1["queries"][1]["rgb_path"]
        assert e0["queries"][1]["rgb_path"] == e1["queries"][0]["rgb_path"]
        for episode in (e0, e1):
            assert [q["present"] for q in episode["queries"]] == [True, False]
            assert episode["queries"][0]["candidate_variant"] == episode["enrolled_variant"]
            assert episode["queries"][1]["candidate_variant"] != episode["enrolled_variant"]


def test_evidence_files_are_write_once(tmp_path):
    file = tmp_path / "evidence.json"
    builder.write_new(file, {"valid":True})
    with pytest.raises(FileExistsError):
        builder.write_new(file, {"valid":False})


def fake_renderer_episode(family_id, object_seed, enrolled_variant, scene_seed, distractor_family_ids, support_size, query_size):
    support_mask=np.zeros((support_size,support_size),dtype=bool)
    support_mask[15:80,15:80]=True
    support_mask[35:40,30+10*enrolled_variant:35+10*enrolled_variant]=False
    support_rgb=np.full((support_size,support_size,3),17,dtype=np.uint8)
    support_rgb[support_mask]=160
    boxes=[[10,10,70,70],[90,10,150,70],[170,10,230,70],[10,100,70,160],[90,100,150,160],[170,100,230,160]]
    queries={}
    for variant in (0,1):
        union=np.zeros((query_size,query_size),dtype=bool)
        for x1,y1,x2,y2 in boxes:union[y1:y2,x1:x2]=True
        candidate=np.zeros_like(union);candidate[10:70,10:70]=True
        candidate[30:35,25+10*variant:30+10*variant]=False
        union[10:70,10:70]=candidate[10:70,10:70]
        rgb=np.full((query_size,query_size,3),17,dtype=np.uint8);rgb[union]=160
        match=variant==enrolled_variant
        records=[{"family_id":fid,"micro_variant":variant if i==0 else 0,"bbox_xyxy":box,
                  "instance_seed":1,"instance_key":"query-instance", "parameters":{"axis_jitter":[1.01,1.02],"connector_gain":1.01}}
                 for i,(fid,box) in enumerate(zip([family_id]+distractor_family_ids,boxes))]
        queries[str(variant)]={"query_rgb":rgb,"union_mask":union,"candidate_mask":candidate,
                              "target_mask":candidate.copy() if match else np.zeros_like(candidate),
                              "candidate_bbox_xyxy":boxes[0],"target_bbox_xyxy":boxes[0] if match else None,
                              "object_records":records,"object_boxes":boxes,"matches_reference":match,
                              "micro_statistics":{"primitive_count":2,"continuous_removed_area":.02,
                                                  "minimum_nominal_feature_width_pixels":5.0}}
    return {"support_rgb":support_rgb,"support_mask":support_mask,"support_bbox_xyxy":[15,15,80,80],"queries":queries,
            "family_spec":{"family_id":family_id,"grammar_digits":builder.family_digits(family_id),
                           "canonical_program_id":"test-program","family_name":"test-family"},
            "pair_statistics":{"query_candidate_is_new_macro_instance":True},
            "parameters":{"support_parameters":{"axis_jitter":[1.,1.],"connector_gain":1.}}}


def test_rendered_evidence_preserves_counterbalanced_pixels_and_target_masks(plan,tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(builder,"DATA",tmp_path)
    monkeypatch.setattr(builder.importlib,"import_module",lambda _:SimpleNamespace(render_episode=fake_renderer_episode))
    family=plan["families"][0]
    report,episodes=builder.render_family((family,plan["episodes"][:2]))
    assert report["counterbalance_exact"] and report["continuous_removed_area_equal"]
    assert len(list(tmp_path.rglob("*.*")))==11
    for episode in episodes:
        for query in episode["queries"]:
            with np.load(tmp_path/query["target_path"]) as targets:
                assert bool(targets["target_mask"].any()) == query["present"]
            assert len(query["all_boxes"])==6
    assert episodes[0]["queries"][0]["rgb_array_sha256"]==episodes[1]["queries"][1]["rgb_array_sha256"]


def test_background_microvariant_shortcuts_are_rejected(plan,tmp_path,monkeypatch):
    from types import SimpleNamespace
    def bad_renderer(*args,**kwargs):
        value=fake_renderer_episode(*args,**kwargs)
        value["queries"]["1"]["query_rgb"][255,255]=[99,99,99]
        return value
    monkeypatch.setattr(builder,"DATA",tmp_path)
    monkeypatch.setattr(builder.importlib,"import_module",lambda _:SimpleNamespace(render_episode=bad_renderer))
    with pytest.raises(ValueError,match="confined to candidate"):
        builder.render_family((plan["families"][0],plan["episodes"][:2]))


def test_wrong_negative_target_annotation_is_rejected(plan,tmp_path,monkeypatch):
    from types import SimpleNamespace
    def bad_renderer(*args,**kwargs):
        value=fake_renderer_episode(*args,**kwargs)
        for query in value["queries"].values():query["target_mask"]=query["candidate_mask"].copy()
        return value
    monkeypatch.setattr(builder,"DATA",tmp_path)
    monkeypatch.setattr(builder.importlib,"import_module",lambda _:SimpleNamespace(render_episode=bad_renderer))
    with pytest.raises(ValueError,match="violates enrolled identity"):
        builder.render_family((plan["families"][0],plan["episodes"][:2]))


def test_real_renderer_and_builder_integrate_on_reserved_only_family(plan,tmp_path,monkeypatch):
    import copy
    reserved=plan["family_partitions"]["reserved"][:6]
    family=copy.deepcopy(plan["families"][0])
    family.update(family_id=reserved[0],grammar_digits=builder.family_digits(reserved[0]),
                  distractor_family_ids=reserved[1:],split="reserved",
                  object_seed=builder.seeded_int("reserved-builder-smoke-object"),
                  scene_seed=builder.seeded_int("reserved-builder-smoke-scene"))
    episodes=copy.deepcopy(plan["episodes"][:2])
    for episode in episodes:
        for key in ("family_id","grammar_digits","distractor_family_ids","split","object_seed","scene_seed"):
            episode[key]=family[key]
    monkeypatch.setattr(builder,"DATA",tmp_path)
    report,rows=builder.render_family((family,episodes))
    assert report["paired_changed_pixels"]>0 and report["pair_changes_candidate_only"]
    assert report["reference_query_macro_parameters_differ"]
    assert rows[0]["queries"][0]["rgb_array_sha256"]==rows[1]["queries"][1]["rgb_array_sha256"]
