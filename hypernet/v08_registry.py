"""Inference ablations with one shared scratch CNN per support image."""
from .models.gaussian_reference import GaussianReferenceRouter
from .models.transport_region import TransportFamily,TransportVariant

MODEL_MAP={'v07_cnn':'baseline','v08_assignment':'assignment','v08_foreground':'foreground',
           'v08_tolerance':'tolerance','v08_counterfactual':'counterfactual','v08_full':'full'}
ALIASES=tuple(MODEL_MAP)+('fixed_v06','fixed_v08')

def make_v08_models(checkpoint_dir,steps=120,seed=73,device='cpu'):
    router=GaussianReferenceRouter(mode='shape_pattern',checkpoint_dir=checkpoint_dir,
                                  steps=steps,seed=seed,device=device)
    learned=TransportFamily(router,device=device)
    fixed=TransportFamily(device=device)
    result={name:TransportVariant(learned,variant) for name,variant in MODEL_MAP.items()}
    result['fixed_v06']=TransportVariant(fixed,'baseline')
    result['fixed_v08']=TransportVariant(fixed,'full')
    return result
