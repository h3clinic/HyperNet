"""Fixed analytic-target control for attributing scratch-CNN improvements.

This matcher replaces only TinyV3Matcher's learned CNN descriptor with the
exact unlearned target used to train that CNN. Resizing, patch scales, point
coordinates and reciprocal matching are held constant. It is a mathematical
image descriptor, not a CNN and not a learned one-shot category model.
"""
from collections import OrderedDict
import math
import numpy as np
import torch

from ..types import MatchResult, rgb_uint8
from ..training_v3 import analytic_shape_target, multiscale_patch_grid
from .cnn import _cached, _local_result, _tensor, image_digest


class AnalyticPatchMatcher:
    name = "analytic_patches"

    def __init__(self, device="cpu", image_size=256, patch_size=32,
                 scale_factors=(1, 2, 4), keep_fraction=.35, cache_size=512):
        if not math.isfinite(keep_fraction) or not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if int(patch_size) != patch_size or patch_size < 8:
            raise ValueError("patch_size must be an integer of at least eight")
        if (not scale_factors or any(int(s) != s or s < 1 for s in scale_factors)
                or len(set(scale_factors)) != len(scale_factors)):
            raise ValueError("scale_factors must be unique positive integers")
        if int(image_size) != image_size or image_size < patch_size * max(scale_factors):
            raise ValueError("image_size must accommodate every slice scale")
        if int(cache_size) != cache_size or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.device = str(device)
        self.image_size, self.patch_size = int(image_size), int(patch_size)
        self.scale_factors = tuple(int(s) for s in scale_factors)
        self.keep_fraction, self.cache_size = float(keep_fraction), int(cache_size)
        self.cache = OrderedDict()

    def provenance(self):
        return {
            "method": self.name,
            "architecture": "fixed 64-D HOG/coarse-gray/edge measurement; no CNN",
            "role": "analytic target control with TinyV3 patch extraction and scoring",
            "is_cnn": False, "learned_parameters": 0, "external_pretraining": False,
            "external_training_images": 0, "target_reference_images": 1,
            "target_weight_updates": 0, "query_weight_updates": 0,
            "handcrafted_cues_not_learned": True, "category_labels_used": False,
            "image_size": self.image_size, "patch_size": self.patch_size,
            "scale_factors": list(self.scale_factors), "embedding_dim": 64,
            "keep_fraction": self.keep_fraction, "cache_size": self.cache_size,
            "feature_coordinates": "zero-based pixel centers",
            "matching": "partial reciprocal cosine matching across multiscale analytic slices",
            "score_is_probability": False, "geometry_verified": False,
            "reference_abstention": "spatially uniform RGB reference; avoids numerical edge artifacts",
            "sources": ["https://arxiv.org/abs/2112.09133"],
            "paper_reproduction": False,
            "limitation": "Fixed shape measurements test the value of CNN learning; they do not learn category semantics.",
        }

    def prepare(self, reference):
        self._features(rgb_uint8(reference))

    def _features(self, image):
        image = rgb_uint8(image)
        key = image_digest(image)
        cached = _cached(self.cache, key)
        if cached is not None:
            return cached
        with torch.inference_mode():
            tensor = _tensor(image, self.image_size, self.device, normalize=False)
            patches, points, _ = multiscale_patch_grid(tensor, self.patch_size, self.scale_factors)
            embedding = torch.cat([analytic_shape_target(patches[start:start + 128]).cpu()
                                   for start in range(0, len(patches), 128)]).numpy()
            points = points.cpu().numpy()
        points[:, 0] = points[:, 0] * image.shape[1] / self.image_size - .5
        points[:, 1] = points[:, 1] * image.shape[0] / self.image_size - .5
        return _cached(self.cache, key, (embedding, points), self.cache_size)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        if not np.any(reference != reference[0, 0]):
            return MatchResult(0., {"method": self.name, "reason": "spatially_uniform_reference",
                                   "is_cnn": False, "learned_parameters": 0,
                                   "external_pretraining": False, "external_training_images": 0,
                                   "target_reference_images": 1, "target_weight_updates": 0,
                                   "query_weight_updates": 0, "handcrafted_cues_not_learned": True,
                                   "score_is_probability": False, "geometry_verified": False})
        a, pa = self._features(reference)
        b, pb = self._features(query)
        return _local_result(
            a, b, pa, pb,
            {"method": self.name, "is_cnn": False, "learned_parameters": 0,
             "external_pretraining": False, "external_training_images": 0,
             "target_reference_images": 1, "target_weight_updates": 0,
             "query_weight_updates": 0, "handcrafted_cues_not_learned": True,
             "reference_descriptors": len(a), "query_descriptors": len(b),
             "slice_scale_factors": self.scale_factors},
            self.keep_fraction,
        )

