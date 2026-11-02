import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'examples/replay_v12.py'
spec=importlib.util.spec_from_file_location('portable_v12_replay',path)
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


def stats():return dict(float_values_checked=0,leaf_values_checked=0,max_absolute_float_error=0.,nonzero_float_differences=0)


def test_default_selection_covers_both_partitions_and_reference_variants_without_scores():
    episodes=[dict(episode_id=f'{split}_{family}_{variant}',split=split,family_id=family,enrolled_variant=variant)
              for split in ('dev','test') for family in (2,1) for variant in (1,0)]
    chosen=replay.select_episodes(episodes)
    assert [e['episode_id'] for e in chosen]==['dev_1_0','test_1_0','dev_1_1','test_1_1']
    assert replay.select_episodes(episodes,full=True)==episodes


@pytest.mark.parametrize('limit',[0,-1,True,3])
def test_invalid_demo_budget_is_rejected(limit):
    with pytest.raises(ValueError):replay.select_episodes([dict(split='dev')],limit)


def test_float_tolerance_is_absolute_and_discrete_fields_remain_exact():
    values=stats();replay.compare_json({'score':.5+1e-7,'box':[1,2,3,4]}, {'score':.5,'box':[1,2,3,4]},1e-6,values)
    assert values['nonzero_float_differences']==1
    with pytest.raises(ValueError,match='Integer'):
        replay.compare_json([1.00000001],[1],1e-6,stats())
    with pytest.raises(ValueError,match='Float mismatch'):
        replay.compare_json(1000000.+.001,1000000.,1e-6,stats())


def test_exact_mode_nonfinite_values_missing_fields_and_boolean_types():
    for actual,expected in [(float('nan'),.5),(float('inf'),.5),(.5+1e-12,.5),({'score':.5},{}),(1,True)]:
        with pytest.raises(ValueError):replay.compare_json(actual,expected,0.,stats())


def test_portable_manifest_allows_only_provenance_path_remapping():
    first={'completed':True,'episodes':[{'id':'a','rgb_sha256':'123'}],'counts':{'episodes':1},'family_partitions':{'dev':[1]},'prior_path':'original'}
    second={**first,'prior_path':'portable'}
    replay.validate_manifest(first,second)
    changed={**first,'episodes':[{'id':'a','rgb_sha256':'456'}]}
    with pytest.raises(ValueError,match='episodes'):replay.validate_manifest(changed,second)


def test_reproduction_cannot_accept_loose_or_nonfinite_tolerance_before_io(tmp_path):
    for tolerance in (-1.,float('nan'),float('inf'),1e-3):
        with pytest.raises(ValueError,match='atol'):
            replay.run(tmp_path,tmp_path/'weights',tmp_path/'output',atol=tolerance)


def test_nonempty_replication_output_is_preserved(tmp_path):
    (tmp_path/'evidence.txt').write_text('keep')
    with pytest.raises(FileExistsError):replay.run(tmp_path,tmp_path/'weights',tmp_path)
    assert (tmp_path/'evidence.txt').read_text()=='keep'
