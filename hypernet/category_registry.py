"""Registry for the category task; keeps historical image-matching aliases intact."""
from .registry import make_models
from .scratch_registry import SCRATCH_MODEL_NAMES
from .v05_registry import V05_MODEL_NAMES

CATEGORY_MODEL_NAMES=("sift","tiny","tiny_v2","tiny_v2_whitened_no_geometry",
                      "category_global","category_regions","category_parts") + SCRATCH_MODEL_NAMES + V05_MODEL_NAMES

def make_category_models(names,*,weights_path=None,allow_download=False,
                         checkpoint_dir="checkpoints/category",tiny_steps=120,
                         seed=53,device="cpu"):
    names=list(dict.fromkeys(names))
    if not names or set(names)-set(CATEGORY_MODEL_NAMES):
        raise ValueError(f"Choose category methods from {CATEGORY_MODEL_NAMES}")
    v05=[n for n in names if n in V05_MODEL_NAMES or
         (n in {'analytic_patches','tiny_v3'} and any(s in V05_MODEL_NAMES for s in names))]
    scratch=[n for n in names if n not in v05 and (n in SCRATCH_MODEL_NAMES or
             (n == "tiny_v2" and any(s in SCRATCH_MODEL_NAMES for s in names)))]
    legacy=[n for n in names if not n.startswith("category_") and n not in scratch and n not in v05]
    pool=make_models(legacy,checkpoint_dir=checkpoint_dir,tiny_steps=tiny_steps,
                     seed=seed,device=device) if legacy else {}
    if scratch:
        from .scratch_registry import make_scratch_models
        pool.update(make_scratch_models(scratch, checkpoint_dir=checkpoint_dir,
                    steps=tiny_steps, seed=seed, device=device))
    if v05:
        from .v05_registry import make_v05_models
        pool.update(make_v05_models(v05,checkpoint_dir=checkpoint_dir,steps=tiny_steps,seed=seed,device=device))
    if any(n.startswith("category_") for n in names):
        from .models.category import CategoryMatcher,ResNetCategoryFeatures
        extractor=ResNetCategoryFeatures(weights_path=weights_path,
            allow_download=allow_download,device=device)
        for name in names:
            if name.startswith("category_"):
                pool[name]=CategoryMatcher(mode=name.removeprefix("category_"),extractor=extractor)
    return {n:pool[n] for n in names}

