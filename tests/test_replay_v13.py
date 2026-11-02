import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'examples/replay_v13.py'
spec=importlib.util.spec_from_file_location('replay_v13',path)
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def test_default_selection_is_test_positive_negative_and_score_independent():
    rows=[{'id':'z','split':'test','present':True,'score':100},
          {'id':'a','split':'dev','present':True},
          {'id':'b','split':'test','present':True,'score':-100},
          {'id':'d','split':'test','present':False}]
    assert [r['id'] for r in replay.select_queries(rows)]==['b','d']
    assert replay.select_queries(rows,True)==rows
    with pytest.raises(ValueError):replay.select_queries(rows[:1])


def test_manifest_requires_exact_roster_but_allows_provenance_path_changes():
    first={'schema_version':'hypernet-real-photo-v13-v1','references':[1],'queries':[2],'categories':[3],'counts':{},'source_path':'old'}
    second={**first,'source_path':'portable'}
    replay.validate_manifest(second,first)
    with pytest.raises(ValueError):replay.validate_manifest({**second,'queries':[4]},first)


def test_tolerance_is_absolute_and_never_allows_nonfinite_or_changed_bools():
    stats={'numeric_values_checked':0,'max_absolute_error':0.}
    replay.compare([1.0000001],[1.],1e-6,stats)
    with pytest.raises(ValueError):replay.compare([1000000.1],[1000000.],1e-6,stats)
    with pytest.raises(ValueError):replay.compare([float('nan')],[1.],1e-6,stats)
    with pytest.raises(ValueError):replay.compare([1],[True],1e-6,stats)


def test_floating_boxes_must_match_exactly_even_inside_tolerance():
    row={'id':'one','category_id':3,'image_id':7,'present':True,'bbox_xyxy':[0.,0.,3.,4.],'proposals':[[0.,0.,3.,4.]],'score':.5}
    changed={**row,'bbox_xyxy':[0.,0.,3.0000001,4.]}
    with pytest.raises(ValueError):replay.compare_row(changed,row,1e-6,{'numeric_values_checked':0,'max_absolute_error':0.})


def test_preserve_existing_replication_evidence_before_loading(tmp_path):
    folder=tmp_path/'run';folder.mkdir();(folder/'proof').write_text('keep')
    with pytest.raises(FileExistsError):replay.run(tmp_path/'data',tmp_path/'weights',folder)
    assert (folder/'proof').read_text()=='keep'


def test_invalid_tolerance_rejected_before_input_access(tmp_path):
    for value in [-1.,.1,float('nan')]:
        with pytest.raises(ValueError):replay.run(tmp_path/'data',tmp_path/'weights',tmp_path/'out',atol=value)
