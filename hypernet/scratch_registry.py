"""One-reference routers for scratch-only category experiments."""
import hashlib
from pathlib import Path
import time

from .registry import ReferenceCNNRouter, ImprovedReferenceCNNRouter

SCRATCH_MODEL_NAMES = ("shape_texture", "tiny_v2_shape", "tiny_v2_regions",
                       "tiny_v3", "tiny_v3_shape", "tiny_v3_regions",
                       "tiny_v2_centered", "tiny_v3_centered", "analytic_patches")


class ShapeReferenceCNNRouter(ReferenceCNNRouter):
    name = "single_image_shape_cnn_v3"

    def prepare(self, reference):
        from . import training_v3, training_v2, training, types
        from .models import tiny_v3, cnn
        key = self._key(reference)
        if key in self.matchers:
            return
        # Include training, feature extraction, and inherited preprocessing in cache identity.
        fingerprint = hashlib.sha256(b"".join(Path(module.__file__).read_bytes()
            for module in (training_v3, training_v2, training, types, tiny_v3, cnn))).hexdigest()[:16]
        path = self.checkpoint_dir / f"v3_{fingerprint}_{key[:20]}_steps{self.steps}_seed{self.seed}.pt"
        started = time.perf_counter()
        if path.exists():
            metadata = {"cached_checkpoint": True, "checkpoint": str(path), "target_images": 1}
        else:
            metadata = training_v3.train_reference_cnn_v3(reference, path, steps=self.steps,
                                                        seed=self.seed, device=self.device)
        matcher = tiny_v3.TinyV3Matcher(path, device=self.device)
        if matcher.metadata["reference_rgb_sha256"] != cnn.image_digest(reference):
            raise ValueError("Cached model does not belong to this reference")
        if matcher.metadata.get("external_training_images") != 0:
            raise ValueError("Scratch router rejects external training")
        self.matchers[key] = matcher
        self.training_records.append({"reference_key": key,
            "elapsed_seconds": time.perf_counter() - started,
            "training": metadata, "model": matcher.provenance()})

    def provenance(self):
        return {**super().provenance(), "training_version": 3,
                "analytic_training_target": "fixed HOG/coarse-gray/edge measurements",
                "learned_teacher": False, "external_training_images": 0, "query_weight_updates": 0}


def make_scratch_models(names, *, checkpoint_dir, steps=120, seed=61, device="cpu"):
    """Shared weights make plain / sliced / hybrid comparisons controlled."""
    from .models.shape_texture import ShapeTextureMatcher, ShapeTextureCNNMatcher
    names = list(dict.fromkeys(names))
    allowed = set(SCRATCH_MODEL_NAMES) | {"tiny_v2"}
    if set(names) - allowed:
        raise ValueError("Unknown scratch category model")
    shape = ShapeTextureMatcher()
    pool = {"shape_texture": shape}
    if "analytic_patches" in names:
        from .models.analytic_patches import AnalyticPatchMatcher
        pool["analytic_patches"] = AnalyticPatchMatcher()
    for version in (2, 3):
        prefix = f"tiny_v{version}"
        if not any(n == prefix or n.startswith(prefix + "_") for n in names):
            continue
        router = (ImprovedReferenceCNNRouter(checkpoint_dir, "v2", steps=steps, seed=seed, device=device)
                  if version == 2 else
                  ShapeReferenceCNNRouter(checkpoint_dir, steps=steps, seed=seed, device=device))
        pool[prefix] = router
        pool[prefix + "_shape"] = ShapeTextureCNNMatcher(base_matcher=router, shape_matcher=shape, cnn_weight=.5)
        pool[prefix + "_regions"] = ShapeTextureCNNMatcher(base_matcher=router, shape_matcher=shape, cnn_weight=1.)
        if prefix + "_centered" in names:
            from .models.scratch_centered import CenteredScratchMatcher
            pool[prefix + "_centered"] = CenteredScratchMatcher(base_matcher=router)
    return {name: pool[name] for name in names}
