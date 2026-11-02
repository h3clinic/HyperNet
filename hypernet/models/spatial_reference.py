"""Fresh one-image CNN descriptors inside the shared structured-region matcher."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch

from ..registry import ReferenceCNNRouter
from ..training_v3 import TinyShapeCNN
from ..training_v5 import FORMAT
from ..types import MatchResult, rgb_uint8
from .cnn import image_digest


class SpatialReferenceMatcher:
    name = "spatial_reference_cnn_v5"

    def __init__(self, checkpoint_path, device="cpu", mode="structured"):
        from .region_shape import RegionShapeMatcher
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Unsupported complete-reference checkpoint format")
        self.metadata = checkpoint["metadata"]
        if (self.metadata.get("external_pretraining") is not False or
                self.metadata.get("external_training_images") != 0 or
                self.metadata.get("target_reference_images") != 1 or
                self.metadata.get("query_weight_updates") != 0 or
                self.metadata.get("learned_teacher") is not False):
            raise ValueError("Spatial CNN requires one reference, no external training or teacher, and no query updates")
        self.device = str(device)
        with torch.random.fork_rng(devices=[]):
            self.model = TinyShapeCNN(64)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(device).requires_grad_(False)
        self.checkpoint_sha256 = sha256(path.read_bytes()).hexdigest()
        self.checkpoint_path = str(path.resolve())
        self.region_matcher = RegionShapeMatcher(descriptor=self._describe,
            provenance={"descriptor": "fresh one-reference TinyShapeCNN",
                        "learned_descriptor_only": True,
                        "external_pretraining": False, "external_training_images": 0,
                        "target_reference_images": 1, "query_weight_updates": 0},
            device=device, mode=mode)

    def _describe(self, tiles):
        if tiles.ndim != 4 or tiles.shape[1:] != (3, 32, 32):
            raise ValueError("Region descriptor needs N x 3 x 32 x 32 tiles")
        with torch.inference_mode():
            return torch.cat([self.model(tiles[start:start + 128].to(self.device))
                              for start in range(0, len(tiles), 128)])

    def provenance(self):
        return {**self.region_matcher.provenance(), **self.metadata,
                "method": self.name, "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_checkpoint_path": self.checkpoint_path,
                "learned_descriptor_only": True, "analytic_branch_at_inference": False,
                "matching": "shared candidate boxes and spatial tile scoring with fixed-feature control"}

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if image_digest(reference) != self.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        if self.metadata.get("insufficient_informative_patches"):
            return MatchResult(0., {"method": self.name,
                "reason": "insufficient_informative_reference_patches",
                "external_pretraining": False, "external_training_images": 0,
                "target_reference_images": 1, "query_weight_updates": 0})
        result = self.region_matcher.predict(reference, query)
        result.evidence.update(method=self.name, learned_descriptor_only=True,
                               checkpoint_sha256=self.checkpoint_sha256,
                               query_weight_updates=0, external_pretraining=False,
                               external_training_images=0, target_reference_images=1)
        return result


class V5ReferenceRouter(ReferenceCNNRouter):
    """One independent scratch CNN for each supplied support RGB array."""
    name = "single_reference_spatial_v5"

    def __init__(self, checkpoint_dir, steps=120, seed=73, device="cpu"):
        super().__init__(checkpoint_dir, steps=steps, seed=seed, device=device)

    def prepare(self, reference):
        from .. import training_v5, training_v4, training_v3, training, types
        from . import region_shape, spatial_reference, cnn
        reference = rgb_uint8(reference)
        key = self._key(reference)
        if key in self.matchers:
            return
        modules = (training_v5, training_v4, training_v3, training, types,
                   region_shape, spatial_reference, cnn)
        fingerprint = sha256(b"".join(Path(module.__file__).read_bytes()
                                    for module in modules)).hexdigest()[:16]
        path = self.checkpoint_dir / f"v5_{fingerprint}_{key[:20]}_steps{self.steps}_seed{self.seed}.pt"
        started = time.perf_counter()
        training = ({"cached_checkpoint": True, "checkpoint": str(path), "target_images": 1}
                    if path.exists() else training_v5.train_reference_shape_v5(
                        reference, path, steps=self.steps, seed=self.seed, device=self.device))
        matcher = SpatialReferenceMatcher(path, device=self.device)
        if matcher.metadata.get("reference_rgb_sha256") != image_digest(reference):
            raise ValueError("Cached CNN weights belong to a different reference")
        self.matchers[key] = matcher
        self.training_records.append({"reference_key": key,
            "elapsed_seconds": time.perf_counter() - started,
            "training": training, "model": matcher.provenance()})

    def provenance(self):
        return {**super().provenance(), "training_version": 5,
                "external_pretraining": False, "query_weight_updates": 0,
                "learned_teacher": False, "learned_descriptor_only": True,
                "objective": "one-image shape restoration; full patch coverage plus canonical region tiles",
                "matching": "same reference-conditioned spatial region search as fixed control"}
