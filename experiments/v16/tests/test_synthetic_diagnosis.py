import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from diagnose_relations import retrieval_metrics,PLAN
from analyze_synthetic_relations import bootstrap


def test_retrieval_ranks_ties_and_chance():
    scores=np.array([[1.,1.,0.],[1.,3.,2.],[3.,2.,1.]])
    metrics,prediction,ranks=retrieval_metrics(scores,np.array([1,1,2]))
    assert prediction.tolist()==[0,1,0]
    assert ranks.tolist()==[2,1,3]
    assert metrics['correct']==1 and metrics['chance_top1']==1/3
    assert np.isclose(metrics['mean_reciprocal_rank'],(1/2+1+1/3)/3)


def test_plan_fixed_final_and_independent_instances():
    assert PLAN['references_per_type']==1
    assert PLAN['query_instances_per_type']==2
    assert len(set(PLAN['query_seed_offsets']))==2
    assert PLAN['arms']==['random','primary_final_4096','no_relation_aux_final_4096','raw_gray_cosine']


def test_paired_cluster_bootstrap_zero_and_bounds():
    indices=np.array([[0,0,0],[1,1,1],[2,2,2],[0,1,2]])
    assert bootstrap(np.zeros(3),indices)=={'estimate':0.0,'ci95':[0.0,0.0]}
    result=bootstrap(np.array([0.,.5,1.]),indices)
    assert result['estimate']==.5
    assert 0<=result['ci95'][0]<.5<result['ci95'][1]<=1
