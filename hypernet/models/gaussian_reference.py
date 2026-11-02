"""Reference-bound Gaussian-training ablations using unchanged region inference."""
from pathlib import Path
from hashlib import sha256
import time
import torch
from ..registry import ReferenceCNNRouter
from ..training_gaussian import FORMAT,MODES
from ..training_v3 import TinyShapeCNN
from ..types import rgb_uint8
from .cnn import image_digest
from .spatial_reference import SpatialReferenceMatcher
from .region_shape import RegionShapeMatcher

class GaussianMatcher(SpatialReferenceMatcher):
    name='gaussian_reference_cnn_v07'
    def __init__(self,path,device='cpu'):
        path=Path(path);data=torch.load(path,map_location='cpu',weights_only=True)
        if data.get('format')!=FORMAT:raise ValueError('Invalid Gaussian checkpoint format')
        self.metadata=data['metadata'];self.device=str(device)
        for key,value in {'external_pretraining':False,'external_training_images':0,'target_reference_images':1,
                          'query_weight_updates':0,'learned_teacher':False}.items():
            if self.metadata.get(key)!=value:raise ValueError('One-image scratch constraints failed: '+key)
        if self.metadata.get('mode') not in MODES:raise ValueError('Unknown checkpoint mode')
        with torch.random.fork_rng(devices=[]):self.model=TinyShapeCNN(64)
        self.model.load_state_dict(data['state_dict'],strict=True)
        self.model.eval().to(device).requires_grad_(False)
        self.checkpoint_sha256=sha256(path.read_bytes()).hexdigest();self.checkpoint_path=str(path.resolve())
        self.region_matcher=RegionShapeMatcher(self._describe,{
            'descriptor':'40,976-parameter one-reference scratch CNN','mode':self.metadata['mode'],
            'external_pretraining':False,'external_training_images':0,'target_reference_images':1,
            'query_weight_updates':0,'learned_teacher':False},device=device,cache_size=24)

class GaussianReferenceRouter(ReferenceCNNRouter):
    name='one_reference_gaussian_router_v07'
    def __init__(self,mode,checkpoint_dir,steps=120,seed=73,device='cpu'):
        if mode not in MODES:raise ValueError('Unknown Gaussian ablation')
        super().__init__(checkpoint_dir,steps,seed,device);self.mode=mode
    def prepare(self,reference):
        from .. import training_gaussian,gaussian_variation,training_v5,training_v4,training_v3,training,types
        from . import gaussian_reference,region_shape,spatial_reference,cnn
        reference=rgb_uint8(reference);key=image_digest(reference)
        if key in self.matchers:return
        modules=(training_gaussian,gaussian_variation,training_v5,training_v4,training_v3,training,types,
                 gaussian_reference,region_shape,spatial_reference,cnn)
        fingerprint=sha256(b''.join(Path(m.__file__).read_bytes() for m in modules)).hexdigest()[:16]
        path=self.checkpoint_dir/f'v07_{self.mode}_{fingerprint}_{key[:20]}_steps{self.steps}_seed{self.seed}.pt'
        started=time.perf_counter()
        training=({'cached_checkpoint':True,'checkpoint':str(path)} if path.exists() else
                  training_gaussian.train_reference_gaussian(reference,path,mode=self.mode,steps=self.steps,seed=self.seed,device=self.device))
        matcher=GaussianMatcher(path,self.device)
        for name,value in {'reference_rgb_sha256':key,'mode':self.mode,'seed':self.seed,'requested_steps':self.steps}.items():
            if matcher.metadata.get(name)!=value:raise ValueError('Cached model mismatch: '+name)
        self.matchers[key]=matcher
        self.training_records.append({'reference_key':key,'training':training,'model':matcher.provenance(),
                                      'elapsed_seconds':time.perf_counter()-started})
    def provenance(self):
        return {**super().provenance(),'mode':self.mode,'external_pretraining':False,
                'query_weight_updates':0,'learned_teacher':False,
                'inference':'Unchanged v06 structured region matching; learned descriptors only',
                'target':'Analytic shape measurements of the same transformed clean geometry'}
