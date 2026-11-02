"""Small adapter registry; unavailable dependencies never become fake scores."""
from pathlib import Path
import hashlib
import time
import numpy as np

MODEL_NAMES = ("orb", "sift", "partgraph", "partgraph_no_geometry", "partgraph_no_coverage",
               "partgraph_no_graph", "cnn_global", "cnn_local", "hybrid", "hybrid_gated", "tiny", "loftr",
               "tiny_whitened", "tiny_whitened_no_geometry", "tiny_v2", "tiny_v2_whitened", "tiny_v2_whitened_no_geometry",
               "alignment", "geometry_rescue", "tiny_v2_cascade")

class ReferenceCNNRouter:
    name = "single_image_patch_cnn"
    def __init__(self, checkpoint_dir, steps=60, seed=17, device="cpu"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.steps, self.seed, self.device = steps, seed, device
        self.matchers, self.training_records = {}, []

    def _key(self, image):
        return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()

    def prepare(self, reference):
        from hypernet.training import train_reference_cnn
        from hypernet.models.cnn import TinyPatchMatcher
        key = self._key(reference)
        if key not in self.matchers:
            path = self.checkpoint_dir / f"{key[:20]}_steps{self.steps}_seed{self.seed}.pt"
            start = time.perf_counter()
            if path.exists():
                metadata = {"cached_checkpoint": True, "checkpoint": str(path), "target_images": 1}
            else:
                metadata = train_reference_cnn(reference, path, steps=self.steps, seed=self.seed,
                                               device=self.device)
            self.matchers[key] = TinyPatchMatcher(path, device=self.device)
            self.training_records.append({"reference_key": key, "elapsed_seconds": time.perf_counter()-start,
                                          "training": metadata, "model": self.matchers[key].provenance()})

    def predict(self, reference, query):
        self.prepare(reference)
        return self.matchers[self._key(reference)].predict(reference, query)

    def provenance(self):
        return {"method": self.name, "external_training_images": 0,
                "target_reference_images_per_model": 1, "steps_per_reference": self.steps,
                "seed": self.seed, "distinct_reference_models": len(self.matchers),
                "adaptation_records": self.training_records,
                "scope": "Reference-specific CNN patch learning; transforms are derived data, not new observations"}

class ImprovedReferenceCNNRouter(ReferenceCNNRouter):
    """Keep reference-only training and inference interventions independently selectable."""
    def __init__(self, checkpoint_dir, variant, steps=120, seed=17, device="cpu"):
        super().__init__(checkpoint_dir,steps,seed,device)
        if variant not in {"whitened","whitened_no_geometry","v2","v2_whitened","v2_whitened_no_geometry"}:
            raise ValueError("Unknown reference CNN intervention")
        self.variant=variant
        self.name="single_image_"+variant

    def prepare(self,reference):
        key=self._key(reference)
        if key in self.matchers:return
        is_v2=self.variant.startswith("v2")
        if is_v2:
            import hypernet.training_v2 as recipe
            recipe_hash=hashlib.sha256(Path(recipe.__file__).read_bytes()).hexdigest()[:12]
            prefix="v2_"+recipe_hash+"_"
        else:prefix=""
        path=self.checkpoint_dir/f"{prefix}{key[:20]}_steps{self.steps}_seed{self.seed}.pt"
        start=time.perf_counter()
        if is_v2:
            from hypernet.training_v2 import train_reference_cnn_v2 as train
            from hypernet.models.tiny_v2 import TinyV2Matcher as Base
        else:
            from hypernet.training import train_reference_cnn as train
            from hypernet.models.cnn import TinyPatchMatcher as Base
        if path.exists():
            metadata={"cached_checkpoint":True,"checkpoint":str(path),"target_images":1}
        else:
            metadata=train(reference,path,steps=self.steps,seed=self.seed,device=self.device)
        base=Base(path,device=self.device)
        if "whitened" in self.variant:
            from hypernet.models.whitened import WhitenedTinyMatcher
            model=WhitenedTinyMatcher(base_matcher=base,device=self.device,
                geometry=not self.variant.endswith("no_geometry"),boundary_centers=not is_v2)
        else:model=base
        self.matchers[key]=model
        self.training_records.append({"reference_key":key,"elapsed_seconds":time.perf_counter()-start,
                                      "training":metadata,"model":model.provenance()})

    def provenance(self):
        result=super().provenance()
        result.update(method=self.name,intervention=self.variant,
                      training_version=2 if self.variant.startswith("v2") else 1,
                      external_training_images=0)
        return result

def make_models(names, *, weights_path=None, allow_download=False, checkpoint_dir="checkpoints",
                tiny_steps=60, seed=17, device="cpu", loftr_checkpoint=None):
    requested = list(dict.fromkeys(names))
    unknown = set(requested)-set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown models {sorted(unknown)}; choose from {MODEL_NAMES}")
    pool = {}
    def get(name):
        if name in pool:
            return pool[name]
        if name in ("orb", "sift"):
            from hypernet.models.classical import ORBMatcher, SIFTMatcher
            model = (ORBMatcher if name == "orb" else SIFTMatcher)()
        elif name.startswith("partgraph"):
            from hypernet.models.partgraph import PartGraphMatcher
            model = PartGraphMatcher(geometry=name != "partgraph_no_geometry",
                                     coverage=name != "partgraph_no_coverage", graph=name != "partgraph_no_graph")
        elif name.startswith("cnn_"):
            from hypernet.models.cnn import FrozenCNNMatcher
            model = FrozenCNNMatcher(mode=name.removeprefix("cnn_"), weights_path=weights_path,
                                     allow_download=allow_download, device=device)
        elif name == "hybrid":
            from hypernet.models.fusion import HybridMatcher
            model = HybridMatcher(get("partgraph"), get("cnn_local"))
        elif name == "hybrid_gated":
            from hypernet.models.gated import GatedHybridMatcher
            model = GatedHybridMatcher(get("partgraph"), get("cnn_local"))
        elif name == "tiny":
            model = ReferenceCNNRouter(checkpoint_dir, steps=tiny_steps, seed=seed, device=device)
        elif name in {"tiny_whitened","tiny_whitened_no_geometry","tiny_v2","tiny_v2_whitened","tiny_v2_whitened_no_geometry"}:
            model = ImprovedReferenceCNNRouter(checkpoint_dir,name.removeprefix("tiny_"),steps=tiny_steps,seed=seed,device=device)
        elif name == "alignment":
            from hypernet.models.alignment import AlignmentMatcher
            model=AlignmentMatcher()
        elif name in {"geometry_rescue","tiny_v2_cascade"}:
            from hypernet.models.alignment import EvidenceFallbackMatcher
            primary=get("partgraph") if name=="geometry_rescue" else get("tiny_v2_whitened")
            model=EvidenceFallbackMatcher(primary,get("alignment"))
        else:
            from hypernet.models.learned_matchers import LoFTRMatcher
            model = LoFTRMatcher(checkpoint_path=loftr_checkpoint, allow_download=allow_download, device=device)
        pool[name] = model
        return model
    return {name: get(name) for name in requested}
