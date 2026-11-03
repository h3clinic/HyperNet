import sys,importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('gravity_evaluator',ROOT/'scripts/evaluate_gravity_spy_v15.py')
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
from hypernet.pattern_matcher import raw_pixel_vector


def test_union_wrong_target_rate_counts_each_screened_event_once():
    point='budget_10_per10000';cutoffs={label:{'available':True,point:{'cutoff':{'threshold':.5}}}for label in('A','B')}
    rows=[]
    for label,group,a,b in[('A','g1',.9,.9),('B','g1',.9,.1),('background','g2',.1,.1)]:
        rows.append({'label':label,'group_id':group,'scores':{'method':{'A':{'score':a},'B':{'score':b}}}})
    result=evaluator.union_event_summary(rows,('A','B'),'method',cutoffs)
    assert result['screened_events']==3 and result['wrong_target_alarm_events']==2
    assert result['wrong_target_alarm_events_per10000_screened_events']==pytest.approx(20000/3)
    assert result['target_recovery_recall']==.5
    assert result['screened_time_groups']==2 and result['time_groups_with_any_wrong_alarm']==1
    assert result['background_events']==1 and result['background_events_with_any_alarm']==0


def test_raw_pixel_control_has_no_weight_fit_and_is_centered():
    rng=np.random.default_rng(3);rgb=rng.integers(20,180,(120,160,3),dtype=np.uint8)
    a=raw_pixel_vector(rgb);b=raw_pixel_vector(rgb+20)
    assert a.shape==(2304,)and np.linalg.norm(a)==pytest.approx(1.,abs=1e-6)
    assert np.dot(a,b)==pytest.approx(1.,abs=1e-6)
    assert np.linalg.norm(raw_pixel_vector(np.full_like(rgb,70)))==0


def test_recipe_selection_is_one_global_recipe_not_per_target():
    m={key:{'macro':{'recall':recall,'auroc':auc}}for key,recall,auc in[
        ('learned:bag',.2,.8),('learned:translation_006',.3,.7),('learned:translation_012',.3,.75)]}
    assert evaluator.select_recipe(m)['selected_recipe']=='translation_012'
