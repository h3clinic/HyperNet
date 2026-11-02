"""Exact analytic information with a bounded scratch-CNN contribution."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import math
from pathlib import Path

import numpy as np
import torch

from ..types import MatchResult, rgb_uint8
from ..training_v4 import FORMAT
from ..training_v3 import TinyShapeCNN, analytic_shape_target, multiscale_patch_grid
from .cnn import _cached, _local_result, _tensor, image_digest


def combine_shape_descriptors(analytic, learned, learned_weight=.2):
    """Keep the exact analytic vector in its own coordinates; never overwrite it.

    For alpha in (0,1), the dot product is (1-alpha)*analytic_dot +
    alpha*learned_dot. With unit-norm (or subunit-norm) branches, each pairwise
    similarity changes by at most 2*alpha. Reciprocal argmax selection can still
    change discontinuously, so no final match-score/recall bound is claimed.
    """
    if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
        raise ValueError("learned_weight must be finite and in [0,1]")
    analytic = np.asarray(analytic, dtype=np.float32)
    if analytic.ndim != 2 or not np.isfinite(analytic).all():
        raise ValueError("Analytic features must be a finite N x D matrix")
    if learned_weight == 0:
        return analytic
    learned = np.asarray(learned, dtype=np.float32)
    if learned.shape != analytic.shape or not np.isfinite(learned).all():
        raise ValueError("Learned features must match the finite analytic matrix shape")
    if learned_weight == 1:
        return learned
    return np.concatenate([math.sqrt(1 - learned_weight) * analytic,
                           math.sqrt(learned_weight) * learned], axis=1)


class GuardedShapeMatcher:
    name = "guarded_shape_cnn_v4"

    def __init__(self, checkpoint_path, learned_weight=.2, device="cpu", image_size=None,
                 keep_fraction=.35, cache_size=256):
        if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
            raise ValueError("learned_weight must be finite and in [0,1]")
        if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0,1]")
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != FORMAT:
            raise ValueError("Unsupported shape restoration checkpoint format")
        self.metadata = checkpoint["metadata"]
        if (self.metadata.get("external_training_images") != 0 or
                self.metadata.get("external_pretraining") is not False or
                self.metadata.get("target_reference_images") != 1):
            raise ValueError("Guarded matching requires one reference and zero external training")
        self.image_size = int(image_size or self.metadata["image_size"])
        self.patch_size = int(self.metadata["patch_size"])
        self.scale_factors = tuple(self.metadata["scale_factors"])
        if self.image_size < self.patch_size * max(self.scale_factors):
            raise ValueError("image_size must accommodate every slice scale")
        self.keep_fraction, self.device = float(keep_fraction), str(device)
        self.learned_weight, self.cache_size = float(learned_weight), cache_size
        # Loading should not perturb a caller's training random stream.
        with torch.random.fork_rng(devices=[]):
            self.model = TinyShapeCNN(64)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.model.requires_grad_(False)
        self.checkpoint_sha256 = sha256(path.read_bytes()).hexdigest()
        self.checkpoint_path = str(path.resolve())
        self.feature_cache = OrderedDict()

    def with_learned_weight(self, learned_weight):
        """Create an ablation sharing EXACT weights and cached branch features."""
        if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
            raise ValueError("learned_weight must be finite and in [0,1]")
        result = object.__new__(type(self))
        result.__dict__ = dict(self.__dict__)
        result.learned_weight = float(learned_weight)
        return result

    @property
    def effective_learned_weight(self):
        # If there was insufficient reference evidence to train, preserve the
        # analytic matcher instead of feeding its descriptors a random branch.
        if self.metadata.get("insufficient_informative_patches") and self.learned_weight < 1:
            return 0.
        return self.learned_weight

    def provenance(self):
        weight = self.effective_learned_weight
        return {**self.metadata, "method": self.name,
                "checkpoint_sha256": self.checkpoint_sha256,
                "resolved_checkpoint_path": self.checkpoint_path,
                "learned_weight": self.learned_weight, "effective_learned_weight": weight,
                "analytic_weight": 1 - weight, "analytic_branch_preserved": weight < 1,
                "learned_only_ablation": weight == 1,
                "embedding_dimension_at_inference": 128 if 0 < weight < 1 else 64,
                "inference_image_size": self.image_size, "keep_fraction": self.keep_fraction,
                "query_weight_updates": 0, "geometry_verified": False,
                "feature_coordinates": "zero-based pixel centers", "score_is_probability": False,
                "combination": "concatenate sqrt(1-alpha)*exact analytic and sqrt(alpha)*CNN; no post-concatenation normalization",
                "pairwise_similarity_perturbation_bound": 2 * weight,
                "bound_scope": "Descriptor inner products only, assuming both branches have norm <=1. Reciprocal matches and final scores have no such guarantee.",
                "low_information_fallback": "analytic only if CNN training had fewer than two informative slices; learned-only ablation abstains",
                "handcrafted_cues_not_learned": weight < 1,
                "matching": "unchanged partial reciprocal appearance matching across the same multiscale slices"}

    def _components(self, image, need_learned=True):
        image = rgb_uint8(image)
        key = image_digest(image)
        cached = _cached(self.feature_cache, key)
        if cached is not None and (not need_learned or cached[1] is not None):
            return cached
        with torch.inference_mode():
            tensor = _tensor(image, self.image_size, self.device, normalize=False)
            patches, points, _ = multiscale_patch_grid(tensor, self.patch_size, self.scale_factors)
            if cached is None:
                analytic = torch.cat([analytic_shape_target(patches[start:start + 128]).cpu()
                                      for start in range(0, len(patches), 128)]).numpy()
                points = points.cpu().numpy()
                points[:, 0] = points[:, 0] * image.shape[1] / self.image_size - .5
                points[:, 1] = points[:, 1] * image.shape[0] / self.image_size - .5
            else:
                analytic, _, points = cached
            learned = (torch.cat([self.model(patches[start:start + 128]).cpu()
                                  for start in range(0, len(patches), 128)]).numpy()
                       if need_learned else None)
        return _cached(self.feature_cache, key, (analytic, learned, points), self.cache_size)

    def _features(self, image):
        weight = self.effective_learned_weight
        analytic, learned, points = self._components(image, need_learned=weight > 0)
        return combine_shape_descriptors(analytic, learned, weight), points

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if image_digest(reference) != self.metadata["reference_rgb_sha256"]:
            raise ValueError("Checkpoint was trained on a different reference image")
        evidence = {"method": self.name, "external_pretraining": False, "external_training_images": 0,
                    "target_reference_images": 1, "query_weight_updates": 0,
                    "learned_weight": self.learned_weight,
                    "effective_learned_weight": self.effective_learned_weight,
                    "analytic_branch_preserved": self.effective_learned_weight < 1,
                    "geometry_verified": False, "score_is_probability": False}
        if not np.any(reference != reference[0, 0]):
            return MatchResult(0., {**evidence, "reason": "spatially_uniform_reference"})
        if self.learned_weight == 1 and self.metadata.get("insufficient_informative_patches"):
            return MatchResult(0., {**evidence, "reason": "insufficient_informative_reference_patches"})
        a, pa = self._features(reference)
        b, pb = self._features(query)
        return _local_result(a, b, pa, pb,
                             {**evidence, "reference_descriptors": len(a), "query_descriptors": len(b),
                              "slice_scale_factors": self.scale_factors}, self.keep_fraction)
