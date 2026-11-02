"""Retain exact analytic measurements alongside a reference-trained descriptor."""
import math
import numpy as np
from ..types import MatchResult, rgb_uint8
from .analytic_patches import AnalyticPatchMatcher
from .cnn import _local_result, image_digest


def protected_descriptor(analytic, learned, learned_weight):
    """A dot-product mixture; do not renormalize away the protected branch.

    For unit input rows the dot product equals (1-a)*analytic_cos+a*learned_cos.
    Therefore it differs from the analytic dot product by at most 2*a.
    This does not bound changes in reciprocal correspondences or final recall.
    """
    if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
        raise ValueError('learned_weight must be in [0,1]')
    a,b=np.asarray(analytic,np.float32),np.asarray(learned,np.float32)
    if a.ndim!=2 or b.ndim!=2 or len(a)!=len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite aligned descriptor rows are required')
    if learned_weight==0:return a.copy()
    if learned_weight==1:return b.copy()
    return np.concatenate([np.sqrt(1-learned_weight)*a,np.sqrt(learned_weight)*b],axis=1)


class AnalyticBypassMatcher:
    name='analytic_bypass_existing_cnn'
    def __init__(self,base_matcher,*,learned_weight=.2,analytic_matcher=None):
        if not math.isfinite(learned_weight) or not 0<=learned_weight<=1:
            raise ValueError('learned_weight must be in [0,1]')
        provenance=base_matcher.provenance()
        if provenance.get('external_training_images')!=0 or provenance.get('external_pretraining',False) not in (False,None):
            raise ValueError('A scratch-only base is required')
        self.base_matcher=base_matcher
        self.learned_weight=float(learned_weight)
        self.analytic=analytic_matcher if analytic_matcher is not None else AnalyticPatchMatcher()

    def _base(self,reference):
        if hasattr(self.base_matcher,'_features'):return self.base_matcher
        return self.base_matcher.matchers[self.base_matcher._key(reference)]

    def prepare(self,reference):
        reference=rgb_uint8(reference)
        if hasattr(self.base_matcher,'prepare'):self.base_matcher.prepare(reference)
        base=self._base(reference)
        if base.metadata.get('reference_rgb_sha256')!=image_digest(reference):
            raise ValueError('Checkpoint belongs to another reference')
        provenance=base.provenance()
        if provenance.get('external_training_images')!=0 or provenance.get('external_pretraining',False) not in (False,None):
            raise ValueError('External training is forbidden')
        self.analytic.prepare(reference)

    def predict(self,reference,query):
        reference,query=rgb_uint8(reference),rgb_uint8(query)
        self.prepare(reference);base=self._base(reference)
        weight=0. if base.metadata.get('insufficient_informative_patches') else self.learned_weight
        if weight==0 or not np.any(reference!=reference[0,0]):
            result=self.analytic.predict(reference,query)
            result.evidence.update(method=self.name,effective_learned_weight=0.,analytic_fallback=True)
            return result
        a,pa=self.analytic._features(reference);b,pb=self.analytic._features(query)
        ca,pca=base._features(reference);cb,pcb=base._features(query)
        if pa.shape!=pca.shape or pb.shape!=pcb.shape or not np.allclose(pa,pca,atol=1e-5,rtol=0) or not np.allclose(pb,pcb,atol=1e-5,rtol=0):
            raise ValueError('Analytic and CNN slices must be aligned')
        return _local_result(protected_descriptor(a,ca,weight),protected_descriptor(b,cb,weight),pa,pb,
            {'method':self.name,'external_training_images':0,'external_pretraining':False,
             'target_reference_images':1,'query_weight_updates':0,'learned_weight':weight,
             'analytic_information_retained':weight<1,'analytic_branch_dimensions':64 if weight<1 else 0,
             'geometry_verified':False,'score_is_probability':False},self.analytic.keep_fraction)

    def provenance(self):
        return {'method':self.name,'external_training_images':0,'external_pretraining':False,
                'target_reference_images':1,'query_weight_updates':0,'learned_weight':self.learned_weight,
                'analytic_information_retained':self.learned_weight<1,
                'handcrafted_cues_not_learned':self.learned_weight<1,
                'fusion':'separate scaled concatenated branches',
                'similarity_perturbation_bound':2*self.learned_weight,
                'bound_scope':'Unit-descriptor dot products only; no guarantee for selected matches or category accuracy',
                'cnn':self.base_matcher.provenance(),'fixed_representation':self.analytic.provenance()}
