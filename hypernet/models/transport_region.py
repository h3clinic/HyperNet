"""Support-derived tolerance and capacity-constrained regional part evidence.

This borrows exact partial linear assignment, not a pretrained graph model.
Counterfactual margins are empirical scores, never calibrated probabilities.
"""
from collections import OrderedDict
from copy import deepcopy
from functools import lru_cache
import numpy as np
from ..types import MatchResult,rgb_uint8
from ..partial_assignment import partial_assignment
from .cnn import image_digest
from .region_shape import RegionShapeMatcher,spatial_region_scores

VARIANTS=('baseline','assignment','foreground','tolerance','counterfactual','full')

@lru_cache(maxsize=2)
def geometry_penalty(n):
    y,x=np.meshgrid(np.arange(n),np.arange(n),indexing='ij')
    p=np.column_stack([x.ravel(),y.ravel()])
    d=np.abs(p[:,None]-p[None,:]).sum(-1)
    return np.where(d<=1,.12*d,2.)

def tolerance_similarity(original,mean,precision,query):
    """Half original cosine, half covariance-weighted cosine. No query fitting."""
    ordinary=np.einsum('id,bjd->bij',original,query,optimize=True)
    numerator=np.einsum('id,bjd->bij',mean*precision,query,optimize=True)
    norm_a=np.sqrt((mean*mean*precision).sum(1)).clip(1e-8)
    norm_b=np.sqrt(np.einsum('id,bjd->bij',precision,query*query,optimize=True)).clip(1e-8)
    weighted=numerator/(norm_a[None,:,None]*norm_b)
    return np.clip(.5*ordinary+.5*weighted,-1,1)

def assignment_region_scores(similarities,weights):
    """Same whole/coarse/fine coefficients as v06, with exact part capacity."""
    scores=[];details=[]
    for sim in similarities:
        coarse=partial_assignment(sim[1:5,1:5],row_weights=weights[1:5],
            min_similarity=.2,min_coverage=.6,geometry_penalty=geometry_penalty(2))
        fine=partial_assignment(sim[5:,5:],row_weights=weights[5:],
            min_similarity=.2,min_coverage=.6,geometry_penalty=geometry_penalty(3))
        # Both levels must explain a material amount of reference evidence.
        valid=coarse['coverage_passed'] and fine['coverage_passed']
        score=(.25*max(0.,float(sim[0,0]))+.25*coarse['score']+.5*fine['score']) if valid else 0.
        scores.append(score);details.append({'coarse':coarse,'fine':fine,'coverage_passed':bool(valid)})
    return np.asarray(scores),details

def fixed_permutations():
    rng=np.random.default_rng(181)
    perms=[]
    for _ in range(4):
        a=rng.permutation(4);b=rng.permutation(9)
        if np.array_equal(a,np.arange(4)):a=np.roll(a,1)
        if np.array_equal(b,np.arange(9)):b=np.roll(b,1)
        perms.append(np.r_[0,1+a,5+b])
    return perms

class TransportFamily:
    """Compute all ablations from exactly the same learned/fixed feature tensors."""
    def __init__(self,base_router=None,device='cpu'):
        self.base_router=base_router;self.device=device
        self.fixed=RegionShapeMatcher(device=device,cache_size=8) if base_router is None else None
        self.statistics={};self.predictions={};self.permutations=fixed_permutations()
    def _region(self,reference):
        if self.base_router is None:return self.fixed
        self.base_router.prepare(reference)
        return self.base_router.matchers[self.base_router._key(reference)].region_matcher
    def prepare(self,reference):
        from ..support_statistics import support_statistics
        reference=rgb_uint8(reference);key=image_digest(reference)
        if key not in self.statistics:
            region=self._region(reference)
            self.statistics[key]=support_statistics(reference,region,samples=16,seed=607)
    def provenance(self,variant):
        base=(self.fixed.provenance() if self.base_router is None else self.base_router.provenance())
        return {**base,'method':'reference_partial_assignment_v08','variant':variant,
            'external_pretraining':False,'external_training_images':0,'target_reference_images':1,
            'query_weight_updates':0,'learned_teacher':False,
            'parts':'Same whole/2x2/3x3 canonical tiles and candidate windows as v06',
            'assignment':'Exact partial linear assignment; capacity1; min_similarity.2; weightedcoverage>=.6 at both levels',
            'tolerance':'16support-only Gaussian variants; diagonal shrinkage; precisioncap; half ordinary/half weighted cosine',
            'foreground':'Support-only heuristic GrabCut/contrast weights; no annotated mask',
            'counterfactual':'Intact score times intact-minus-mean-four-shuffles in each region, then max over regions/flips',
            'score_is_probability':False,'formal_null_test':False,
            'novelty':'Experimental combination of established components; not a reproduction or proven novelty claim'}
    def predict_all(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        self.prepare(reference);key=image_digest(reference)
        pair=(key,image_digest(query))
        if pair in self.predictions:return self.predictions[pair]
        stats=self.statistics[key]
        if stats['abstain']:
            results={v:MatchResult(0.,{'reason':'spatially_uniform_reference','variant':v}) for v in VARIANTS}
            self.predictions[pair]=results;return results
        region=self._region(reference)
        q,boxes=region._query(query);original=region._reference(reference)
        arrays={v:[] for v in VARIANTS};detail_sets={v:[] for v in VARIANTS}
        for orientation in (0,1):
            a=original[orientation];w=stats['part_weights'][orientation]
            sim=np.einsum('id,bjd->bij',a,q,optimize=True)
            tol=tolerance_similarity(a,stats['mu'][orientation],stats['precision'][orientation],q)
            arrays['baseline'].append(spatial_region_scores(a,q,'structured'))
            detail_sets['baseline'].append([None]*len(q))
            for v,s,weight in [('assignment',sim,np.ones(14)),('foreground',sim,w),('tolerance',tol,np.ones(14))]:
                values,details=assignment_region_scores(s,weight)
                arrays[v].append(values);detail_sets[v].append(details)
            full_values,full_details=assignment_region_scores(tol,w)
            for v,s,weight,intact,details in [
                ('counterfactual',sim,np.ones(14),arrays['assignment'][-1],detail_sets['assignment'][-1]),
                ('full',tol,w,full_values,full_details)]:
                nulls=[]
                for permutation in self.permutations:
                    values,_=assignment_region_scores(s[:,permutation,:],weight[permutation])
                    nulls.append(values)
                null=np.mean(nulls,axis=0)
                margin=intact-null
                arrays[v].append(intact*margin)
                detail_sets[v].append([{**d,'intact_score':float(r),'counterfactual_mean':float(n),
                    'counterfactual_margin':float(m)} for d,r,n,m in zip(details,intact,null,margin)])
        results={}
        for variant in VARIANTS:
            values=np.stack(arrays[variant]);orientation,best=np.unravel_index(np.argmax(values),values.shape)
            details=detail_sets[variant][orientation][best]
            evidence={'variant':variant,'reference_flipped':bool(orientation),'candidate_regions':len(boxes),
                'external_pretraining':False,'external_training_images':0,'query_weight_updates':0,
                'target_reference_images':1,'score_is_probability':False,'geometry_verified':False,
                'support_statistics':stats['evidence'],'matching':details}
            result=MatchResult(float(values[orientation,best]),evidence,bbox_xyxy=[float(t) for t in boxes[best]])
            if details is not None:
                rows,cols=details['fine']['rows'],details['fine']['cols']
                x0,y0,x1,y1=boxes[best]
                result.points_reference=[[(1-(int(i)%3+.5)/3 if orientation else (int(i)%3+.5)/3)*reference.shape[1],
                                          (int(i)//3+.5)/3*reference.shape[0]] for i in rows]
                result.points_query=[[x0+(int(j)%3+.5)/3*(x1-x0),y0+(int(j)//3+.5)/3*(y1-y0)] for j in cols]
            results[variant]=result
        self.predictions[pair]=results
        return results

class TransportVariant:
    def __init__(self,family,variant):
        if variant not in VARIANTS:raise ValueError('Unknown transport ablation')
        self.family,self.variant=family,variant
    def prepare(self,reference):self.family.prepare(reference)
    def predict(self,reference,query):return deepcopy(self.family.predict_all(reference,query)[self.variant])
    def provenance(self):return self.family.provenance(self.variant)
