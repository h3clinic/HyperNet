import numpy as np
from hypernet.models.transport_region import (TransportFamily,tolerance_similarity,
    assignment_region_scores,fixed_permutations)
from hypernet.models.cnn import image_digest

def test_tolerance_reduces_to_cosine_when_identity_metric():
    rng=np.random.default_rng(7);a=rng.normal(size=(14,64));a/=np.linalg.norm(a,axis=1,keepdims=True)
    b=rng.normal(size=(3,14,64));b/=np.linalg.norm(b,axis=2,keepdims=True)
    assert np.allclose(tolerance_similarity(a,a,np.ones_like(a),b),np.einsum('id,bjd->bij',a,b))

def test_repeated_patch_cannot_satisfy_the_whole_subject():
    similarity=np.zeros((1,14,14));similarity[:,:,0]=1
    # In both spatial levels just one query part is a good match to all parts.
    similarity[:,1:5,1]=1;similarity[:,5:,5]=1
    score,details=assignment_region_scores(similarity,np.ones(14))
    assert score[0]==0 and not details[0]['coverage_passed']
    assert len(details[0]['fine']['cols'])==1

def test_full_counterfactual_prefers_intact_structure():
    rng=np.random.default_rng(73)
    reference=rng.integers(0,256,(24,24,3),dtype=np.uint8)
    features=np.eye(64,dtype=np.float32)[:14]
    permutations=fixed_permutations()
    class Region:
        def _reference(self,_):return np.stack([features,features])
        def _query(self,_):return np.stack([features,features[permutations[0]]]),[(0,0,12,12),(12,0,24,12)]
    family=TransportFamily();family.fixed=Region()
    family.statistics[image_digest(reference)]={'abstain':False,'mu':np.stack([features,features]),
        'precision':np.ones((2,14,64)),'part_weights':np.ones((2,14)),'evidence':{'synthetic_test':True}}
    results=family.predict_all(reference,reference)
    assert results['full'].bbox_xyxy==[0.,0.,12.,12.]
    assert results['full'].score>0
    assert results['assignment'].score==1
    assert len(set(results['full'].evidence['matching']['fine']['cols']))==9
    assert len(results['full'].points_reference)==9
    assert all(-1<=r.score<=1 for r in results.values())

def test_uniform_support_does_not_create_positive_evidence():
    reference=np.zeros((16,16,3),np.uint8)
    family=TransportFamily()
    family.statistics[image_digest(reference)]={'abstain':True}
    assert all(r.score==0 for r in family.predict_all(reference,reference).values())
