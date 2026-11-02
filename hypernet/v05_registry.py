"""Controlled one-reference comparisons for the shape-preserving v05 experiment."""
import hashlib,time
from pathlib import Path
from .registry import ReferenceCNNRouter
from .scratch_registry import ShapeReferenceCNNRouter

V05_MODEL_NAMES=('shape_bypass_v3','shape_learned_v4','shape_guarded_v4','shape_light_v4','shape_score_v4')


class GuardedReferenceRouter(ReferenceCNNRouter):
    name='single_reference_shape_v4'
    def prepare(self,reference):
        from . import training_v4,training_v3,training_v2,training,types
        from .models import guarded_shape,cnn
        key=self._key(reference)
        if key in self.matchers:return
        fingerprint=hashlib.sha256(b''.join(Path(m.__file__).read_bytes() for m in
            (training_v4,training_v3,training_v2,training,types,guarded_shape,cnn))).hexdigest()[:16]
        path=self.checkpoint_dir/f'v4_{fingerprint}_{key[:20]}_steps{self.steps}_seed{self.seed}.pt'
        start=time.perf_counter()
        metadata=({'cached_checkpoint':True,'checkpoint':str(path),'target_images':1} if path.exists() else
            training_v4.train_reference_shape_v4(reference,path,steps=self.steps,seed=self.seed,device=self.device))
        matcher=guarded_shape.GuardedShapeMatcher(path,learned_weight=1.,device=self.device)
        if matcher.metadata.get('reference_rgb_sha256')!=cnn.image_digest(reference):
            raise ValueError('Cached v4 weights do not belong to reference')
        if matcher.metadata.get('external_training_images')!=0:
            raise ValueError('External training is forbidden')
        self.matchers[key]=matcher
        self.training_records.append({'reference_key':key,'elapsed_seconds':time.perf_counter()-start,
                                      'training':metadata,'model':matcher.provenance()})

    def provenance(self):
        return {**super().provenance(),'training_version':4,'objective':'same-geometry analytic restoration',
                'query_weight_updates':0,'learned_teacher':False}


def make_v05_models(names,*,checkpoint_dir,steps=120,seed=73,device='cpu'):
    from .models.analytic_patches import AnalyticPatchMatcher
    from .models.shape_bypass import AnalyticBypassMatcher
    names=list(dict.fromkeys(names))
    if set(names)-set(V05_MODEL_NAMES)-{'analytic_patches','tiny_v3'}:
        raise ValueError('Unknown shape preservation method')
    analytic=AnalyticPatchMatcher(device=device)
    pool={'analytic_patches':analytic}
    if any(n in names for n in ('tiny_v3','shape_bypass_v3')):
        v3=ShapeReferenceCNNRouter(checkpoint_dir,steps=steps,seed=seed,device=device)
        pool['tiny_v3']=v3
        pool['shape_bypass_v3']=AnalyticBypassMatcher(v3,learned_weight=.2,analytic_matcher=analytic)
    if any(n.startswith('shape_') and n.endswith('_v4') for n in names):
        v4=GuardedReferenceRouter(checkpoint_dir,steps=steps,seed=seed,device=device)
        pool['shape_learned_v4']=v4
        pool['shape_guarded_v4']=AnalyticBypassMatcher(v4,learned_weight=.2,analytic_matcher=analytic)
        pool['shape_light_v4']=AnalyticBypassMatcher(v4,learned_weight=.05,analytic_matcher=analytic)
        if 'shape_score_v4' in names:
            from .models.score_guard import AnalyticScoreGuard
            pool['shape_score_v4']=AnalyticScoreGuard(v4,learned_weight=.2,analytic_matcher=analytic)
    return {n:pool[n] for n in names}
