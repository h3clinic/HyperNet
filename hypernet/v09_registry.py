"""Equal inference geometry, explicit procedural-pretraining controls."""
from pathlib import Path
from .models.gaussian_reference import GaussianReferenceRouter
from .models.procedural_reference import FrozenProceduralMatcher,ProceduralReferenceRouter
from .models.region_shape import RegionShapeMatcher

ALIASES=('scratch_reference','procedural_restoration_frozen','procedural_restoration_adapted',
         'procedural_detail_frozen','procedural_detail_adapted','random_frozen','fixed_shape')

def make_v09_models(checkpoint_dir,pretraining_dir,steps=120,seed=73,device='cpu'):
    checkpoint_dir,pretraining_dir=Path(checkpoint_dir),Path(pretraining_dir)
    restoration=pretraining_dir/'procedural_restoration.pt'
    detail=pretraining_dir/'procedural_detail_metric.pt'
    return {
      'scratch_reference':GaussianReferenceRouter('shape_pattern',checkpoint_dir/'scratch',steps,seed,device),
      'procedural_restoration_frozen':FrozenProceduralMatcher(restoration,device=device),
      'procedural_restoration_adapted':ProceduralReferenceRouter(restoration,checkpoint_dir/'restoration',steps,seed,device),
      'procedural_detail_frozen':FrozenProceduralMatcher(detail,device=device),
      'procedural_detail_adapted':ProceduralReferenceRouter(detail,checkpoint_dir/'detail',steps,seed,device),
      'random_frozen':FrozenProceduralMatcher(None,device=device,random_seed=seed),
      'fixed_shape':RegionShapeMatcher(device=device)}
