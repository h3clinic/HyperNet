from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from hypernet.pattern_matcher import PatternMatcher,select_anchors,RECIPES


def features(n=4):
    y,x=np.meshgrid(np.linspace(.15,.85,n),np.linspace(.15,.85,n),indexing='ij')
    descriptors=np.eye(24,dtype=np.float32)[:n*n].reshape(n,n,24)
    return {'descriptors':descriptors,'detail':descriptors[...,:4].copy(),
        'strength':np.ones((n,n),np.float32),'xy':np.stack((x,y),-1).astype(np.float32),
        'source_shape':[64,64]}


class Dummy:
    device='cpu'
    def encode(self,rgb,sampling_stride=4):return features()
    def provenance(self):return {'natural_weight_training_images':0}


def test_identity_and_translation_consistency_are_high():
    f=features();matcher=PatternMatcher(Dummy());bank=matcher.enroll_features(f)
    assert matcher.score(bank,f)['score']==pytest.approx(1.)
    shifted={k:v.copy()if isinstance(v,np.ndarray)else v for k,v in f.items()}
    shifted['xy']+=.1
    assert matcher.score(bank,shifted)['score']==pytest.approx(1.)


def test_bag_control_does_not_claim_geometric_consistency():
    f=features();matcher=PatternMatcher(Dummy());bank=matcher.enroll_features(f)
    query={k:v.copy()if isinstance(v,np.ndarray)else v for k,v in f.items()}
    query['xy']=1-query['xy']
    result=matcher.score_many([bank],query,recipes=RECIPES)[0]
    assert result['bag']['score']==pytest.approx(1.)
    assert result['translation_006']['score']<.5


def test_single_patch_cannot_supply_full_bank_score():
    f=features();matcher=PatternMatcher(Dummy());bank=matcher.enroll_features(f)
    query={k:v.copy()if isinstance(v,np.ndarray)else v for k,v in f.items()}
    query['descriptors'][:]=0;query['descriptors'][0,0]=f['descriptors'][0,0]
    result=matcher.score(bank,query)
    assert result['score']<.2
    assert result['evidence']['contributing_anchors']==1


def test_zero_detail_abstains_and_no_mutation():
    f=features();matcher=PatternMatcher(Dummy());before={k:v.copy()for k,v in f.items()if isinstance(v,np.ndarray)}
    bank=matcher.enroll_features(f);matcher.score(bank,f)
    assert all(np.array_equal(v,f[k])for k,v in before.items())
    f['strength'][:]=0
    assert matcher.score(bank,f)['score']==0


def test_anchor_roster_is_spatial_and_deterministic():
    f=features();a,c=select_anchors(f);b,d=select_anchors(f)
    assert np.array_equal(a,b)and np.array_equal(c,d)
    assert len(np.unique(c))==16
    f['xy'][0,0]=np.nan
    with pytest.raises(ValueError):select_anchors(f)


def test_vectorized_matches_original_scalar_mechanism():
    from pattern_matcher_scalar_reference import PatternMatcher as Scalar
    rng=np.random.default_rng(1518)
    def random_features(n):
        y,x=np.meshgrid(np.linspace(.01,.99,n),np.linspace(.01,.99,n),indexing='ij')
        return {'descriptors':rng.normal(size=(n,n,24)).astype(np.float32),
            'detail':rng.normal(size=(n,n,4)).astype(np.float32),'strength':rng.uniform(size=(n,n)).astype(np.float32),
            'xy':np.stack((x,y),-1).astype(np.float32),'source_shape':[n*4,n*4]}
    matcher=PatternMatcher(Dummy());scalar=Scalar(Dummy())
    banks=[scalar.enroll_features(random_features(n))for n in(4,7,10)]
    query=random_features(18)
    for kind in('learned','analytic'):
        expected=scalar.score_many(banks,query,recipes=RECIPES,feature_kind=kind)
        actual=matcher.score_many(banks,query,recipes=RECIPES,feature_kind=kind)
        for a,b in zip(actual,expected):
            for recipe in RECIPES:
                assert a[recipe]['score']==pytest.approx(b[recipe]['score'],abs=2e-6)
                assert a[recipe]['evidence']==b[recipe]['evidence']
