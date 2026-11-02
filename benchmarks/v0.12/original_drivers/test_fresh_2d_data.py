from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/prepare_fresh_2d_v12.py'
spec=importlib.util.spec_from_file_location('fresh2d_builder',path)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


@pytest.fixture(scope='module')
def plan():return builder.make_plan()


def test_fresh_families_exclude_all_old_splits_preview_and_smoke(plan):
    old,exposure=builder.prior_exposure()
    chosen={f['family_id'] for f in plan['families']}
    assert len(chosen)==256 and len(exposure['excluded_family_ids'])==29
    assert chosen<=set(old['family_partitions']['reserved'])
    assert not chosen&set(exposure['excluded_family_ids'])
    assert not chosen&{f['family_id'] for f in old['families']}
    assert plan['counts']=={'families':256,'episodes':512,'support_images':512,'query_rows':1024,
                           'unique_query_images':512,'unique_rgb_images':1024,'stored_files':2816,'training_rows':0,
                           'episodes_by_split':{'dev':256,'test':256}}


def test_hash_ranking_has_fixed_reproducible_selection_and_full_factor_coverage(plan):
    old,exposure=builder.prior_exposure()
    eligible=set(old['family_partitions']['reserved'])-set(exposure['excluded_family_ids'])
    ranked=sorted(eligible,key=lambda f:sha256((builder.SPLIT_SALT+str(f)).encode()).digest())
    assert plan['family_partitions']['dev']==sorted(ranked[:128])
    assert plan['family_partitions']['test']==sorted(ranked[128:256])
    assert len(plan['family_partitions']['reserved'])==1507
    for axes in plan['factor_level_counts'].values():
        assert all(set(counts)==set(range(8)) and sum(counts.values())==128 for counts in axes.values())


def test_no_training_rows_and_every_distractor_is_fresh_within_its_partition(plan):
    for family in plan['families']:
        assert family['split'] in ('dev','test')
        assert family['family_id'] not in family['distractor_family_ids']
        assert len(set(family['distractor_family_ids']))==5
        assert set(family['distractor_family_ids'])<=set(plan['family_partitions'][family['split']])
    old,_=builder.prior_exposure()
    for key in ('object_seed','scene_seed'):
        assert not {f[key] for f in plan['families']}&{f[key] for f in old['families']}


def test_same_scene_counterbalance_is_preserved_without_rendering_any_fresh_frame(plan):
    for first,second in zip(plan['episodes'][::2],plan['episodes'][1::2]):
        assert first['family_id']==second['family_id']
        assert first['object_seed']==second['object_seed'] and first['scene_seed']==second['scene_seed']
        assert first['enrolled_variant']==0 and second['enrolled_variant']==1
        assert first['queries'][0]['rgb_path']==second['queries'][1]['rgb_path']
        assert first['queries'][1]['rgb_path']==second['queries'][0]['rgb_path']
        assert first['support']['rgb_path']!=second['support']['rgb_path']


def test_missing_model_protocol_freeze_blocks_generation_before_any_image_write(tmp_path,monkeypatch):
    monkeypatch.setattr(builder,'DATA',tmp_path)
    with pytest.raises(ValueError,match='Explicit model/protocol'):
        builder.admit_generation(None,None)
    with pytest.raises(ValueError,match='missing or changed'):
        builder.admit_generation(tmp_path/'absent.json','abc')
    assert not list(tmp_path.rglob('*.png'))


def test_incomplete_model_selection_or_wrong_data_plan_cannot_admit_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(builder,'DATA',tmp_path)
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'model_selection_complete':False}))
    with pytest.raises(ValueError,match='Model selection must be complete'):
        builder.admit_generation(protocol,builder.digest(protocol))
    protocol.write_text(json.dumps({'model_selection_complete':True,'fresh_test_generated_before_freeze':False,'fresh_plan_sha256':'wrong'}))
    (tmp_path/'plan.json').write_text('{}')
    with pytest.raises(ValueError,match='bind this fresh plan'):
        builder.admit_generation(protocol,builder.digest(protocol))
    assert not list(tmp_path.rglob('*.png'))


def admitted_fixture(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    (data/'plan.json').write_text('{}')
    freeze=tmp_path/'freeze.json';freeze.write_text('{}')
    source=tmp_path/'scorer.py';source.write_text('value=1\n')
    protocol=tmp_path/'protocol.json'
    protocol.write_text(json.dumps({'model_selection_complete':True,'fresh_test_generated_before_freeze':False,
                                   'fresh_plan_sha256':builder.digest(data/'plan.json'),
                                   'behavior_files':{str(source):builder.digest(source)}}))
    monkeypatch.setattr(builder,'DATA',data);monkeypatch.setattr(builder,'FREEZE',freeze)
    monkeypatch.setattr(builder,'ADMISSION',tmp_path/'admission.json')
    return data,source,protocol


def test_admission_binds_selected_scoring_code_and_detects_later_changes(tmp_path,monkeypatch):
    data,source,protocol=admitted_fixture(tmp_path,monkeypatch)
    record=builder.admit_generation(protocol,builder.digest(protocol))
    assert record['no_fresh_images_before_admission'] and builder.check_admission()==record
    assert not list(data.rglob('*.png'))
    source.write_text('value=2\n')
    with pytest.raises(ValueError,match='behavior changed'):
        builder.check_admission()


def test_existing_fresh_pixels_prevent_retroactive_admission(tmp_path,monkeypatch):
    data,source,protocol=admitted_fixture(tmp_path,monkeypatch)
    (data/'unexpected.png').write_bytes(b'prior-image-marker')
    with pytest.raises(ValueError,match='images already exist'):
        builder.admit_generation(protocol,builder.digest(protocol))
    assert not builder.ADMISSION.exists()
