"""Shared inference for controlled fresh-photo v06 comparisons."""
from copy import deepcopy
from .models.cnn import image_digest
from .models.region_shape import RegionShapeMatcher
from .models.spatial_reference import V5ReferenceRouter
from .v05_registry import make_v05_models

ALIASES=('analytic_patches','tiny_v3','shape_score_v4','region_full',
         'region_unordered','region_fixed','region_learned','region_hybrid')


class MemoizedMatcher:
    def __init__(self, base):
        self.base, self.predictions = base, {}
    def prepare(self, reference):
        if hasattr(self.base,'prepare'):
            self.base.prepare(reference)
    def predict(self, reference, query):
        key=(image_digest(reference),image_digest(query))
        if key not in self.predictions:
            self.predictions[key]=self.base.predict(reference,query)
        return deepcopy(self.predictions[key])
    def provenance(self):
        return {**self.base.provenance(),'prediction_memoization':True,
                'latency_note':'Cache reuse is enabled; timings are not a model-speed benchmark'}


class RegionFamily:
    def __init__(self, device='cpu'):
        self.fixed=RegionShapeMatcher(device=device,cache_size=8)
        self.variants={mode:self.fixed.with_mode(mode) for mode in ('full','unordered','structured')}
        self.predictions={}
    def prepare(self, reference):
        self.fixed.prepare(reference)
    def predict(self, mode, reference, query):
        key=(image_digest(reference),image_digest(query))
        if key not in self.predictions:
            self.predictions[key]={name:model.predict(reference,query)
                                   for name,model in self.variants.items()}
        return deepcopy(self.predictions[key][mode])


class RegionVariant:
    def __init__(self,family,mode):self.family,self.mode=family,mode
    def prepare(self,reference):self.family.prepare(reference)
    def predict(self,reference,query):return self.family.predict(self.mode,reference,query)
    def provenance(self):return self.family.variants[self.mode].provenance()


class RegionScoreBlend:
    """Explicit hybrid; bounded scalar fusion without changing either branch."""
    def __init__(self,fixed,learned,weight=.2):
        if not 0<=weight<=1:raise ValueError('Weight must be in [0,1]')
        self.fixed,self.learned,self.weight=fixed,learned,weight
    def prepare(self,reference):
        self.fixed.prepare(reference);self.learned.prepare(reference)
    def predict(self,reference,query):
        a,b=self.fixed.predict(reference,query),self.learned.predict(reference,query)
        result=deepcopy(a)
        result.score=(1-self.weight)*a.score+self.weight*b.score
        result.evidence={**result.evidence,'method':'region_score_hybrid',
            'analytic_score':a.score,'learned_score':b.score,'learned_weight':self.weight,
            'learned_bbox_xyxy':b.bbox_xyxy,'reported_box':'fixed branch; branches need not select the same region',
            'external_training_images':0,'external_pretraining':False,'target_reference_images':1,
            'query_weight_updates':0,'learned_teacher':False}
        return result
    def provenance(self):
        return {'method':'region_score_hybrid','external_training_images':0,
                'external_pretraining':False,'target_reference_images':1,
                'query_weight_updates':0,'learned_teacher':False,'learned_weight':self.weight,
                'fixed_branch':self.fixed.provenance(),'learned_branch':self.learned.provenance(),
                'score':'0.8 fixed structured region score + 0.2 scratch-CNN structured region score',
                'limitation':'Hybrid with a hand-coded shape prior; not a learned-only CNN'}


def make_v06_models(checkpoint_dir,steps=120,seed=73,device='cpu',training_variant='balanced'):
    if training_variant not in {'balanced','uniform297'}:
        raise ValueError('Unknown v5 training schedule')
    old=make_v05_models(['analytic_patches','tiny_v3','shape_score_v4'],
                       checkpoint_dir=checkpoint_dir,steps=steps,seed=seed,device=device)
    family=RegionFamily(device)
    fixed=RegionVariant(family,'structured')
    router=V5ReferenceRouter
    if training_variant=='uniform297':
        from .models.spatial_uniform import UniformReferenceRouter
        router=UniformReferenceRouter
    learned=MemoizedMatcher(router(checkpoint_dir,steps=steps,seed=seed,device=device))
    return {**old,'region_full':RegionVariant(family,'full'),
            'region_unordered':RegionVariant(family,'unordered'),'region_fixed':fixed,
            'region_learned':learned,'region_hybrid':RegionScoreBlend(fixed,learned)}
