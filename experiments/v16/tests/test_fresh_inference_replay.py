import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('fresh_replay_for_test',ROOT/'scripts/replay_fresh_inference.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def policy():return {'query_count':2,'new_cnn_batch_size':4,'replay_tolerance':0,'batch_context_policy':'first_original_batch','context_queries_scored':False,'methods':list(replay.evaluation.ALIASES)}


def test_first_two_in_original_order_and_two_context_images():
    plan={'events':[{'event_id':x}for x in ['z','excluded','a','n','b','m']]}
    effective,queries,context=replay.select_context(plan,{'eligible_event_ids':['a','b','m','n','z']},policy())
    assert [r['event_id']for r in queries]==['z','a']
    assert [r['event_id']for r in context]==['z','a','n','b']
    assert len(effective)==5


def test_small_tail_uses_only_actual_first_batch_and_rejects_unamended_policy():
    plan={'events':[{'event_id':x}for x in ['a','b','c']]};quality={'eligible_event_ids':['c','a','b']}
    _,queries,context=replay.select_context(plan,quality,policy())
    assert len(queries)==2 and len(context)==3
    invalid=policy();invalid.pop('batch_context_policy')
    with pytest.raises(ValueError,match='context'):replay.select_context(plan,quality,invalid)


def test_zero_tolerance_detects_one_float_ulp():
    expected=np.ones((5,2,14));actual=expected.copy()
    assert replay.compare_exact(expected,actual)['exact_values']==140
    actual[2,1,7]=np.nextafter(1.,np.inf)
    result=replay.compare_exact(expected,actual)
    assert not result['passed']and result['exact_values']==139
    assert result['mismatches'][0]['method_index']==2 and result['maximum_absolute_error']>0


def test_resolve_rejects_existing_scores_before_metadata_or_pixel_reads(tmp_path,monkeypatch):
    run=tmp_path/'test';run.mkdir();(run/'event_scores.jsonl').write_text('partial')
    def forbidden(*args,**kwargs):raise AssertionError('Gate/pixels must not be reached')
    monkeypatch.setattr(replay,'metadata_gate',forbidden)
    with pytest.raises(ValueError,match='before any TEST'):replay.resolve(SimpleNamespace(test_run=run))


def test_changed_resolution_binding_rejected_before_model_or_pixels(tmp_path,monkeypatch):
    bound=tmp_path/'source.py';bound.write_text('modified')
    resolution=tmp_path/'resolution.json';resolution.write_text(json.dumps({'bindings':{str(bound):'0'*64}}))
    def forbidden(*args,**kwargs):raise AssertionError('No metadata/pixel/model access after failed binding')
    monkeypatch.setattr(replay,'metadata_gate',forbidden)
    with pytest.raises(ValueError,match='Frozen artifact changed'):
        replay.replay(SimpleNamespace(output=tmp_path/'report.json',resolution=resolution))


def test_completed_gate_rejects_partial_run_without_loading_scores(tmp_path):
    with pytest.raises(ValueError,match='Completed TEST'):
        replay.completed_scores(tmp_path,{'completed':False,'stage':'test'},[],{})
