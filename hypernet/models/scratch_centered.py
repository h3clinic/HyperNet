"""Reference-only centering/shrinkage for scratch CNN regional descriptors.

This method fits a small normalization transform to the ONE support image, not
to any query, query collection or labels. It adds no hand-crafted image scores.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import numpy as np

from ..types import MatchResult, rgb_uint8
from .cnn import image_digest
from .shape_texture import region_boxes, _cnn_region_evidence
from .whitened import fit_reference_whitener, apply_reference_whitener


def fit_reference_diagonal(features, shrinkage=.5, variance_floor=1e-6):
    """Shrink reference-only diagonal variance toward its scalar average.

    This removes a common descriptor offset without fitting a high-dimensional
    correlation matrix to one photograph. It is not population normalization.
    """
    features = np.asarray(features, np.float64)
    if features.ndim != 2 or len(features) < 2 or features.shape[1] < 1:
        raise ValueError("Need at least two nonempty reference descriptors")
    if not np.isfinite(features).all():
        raise ValueError("Reference descriptors must be finite")
    if not math.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be finite and in (0,1]")
    if not math.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    mean = features.mean(axis=0)
    variance = np.square(features - mean).sum(axis=0) / (len(features) - 1)
    mean_variance = float(variance.mean())
    floor = max(variance_floor, mean_variance * 1e-4)
    regularized = np.maximum((1 - shrinkage) * variance + shrinkage * mean_variance, floor)
    transform = np.diag(1 / np.sqrt(regularized))
    return mean, transform, {
        "mode": "diagonal", "shrinkage": float(shrinkage),
        "reference_descriptor_count": len(features), "descriptor_dimension": features.shape[1],
        "mean_variance": mean_variance, "variance_floor": floor,
        "regularized_condition_number": float(regularized.max() / regularized.min()),
        "fit_observations": "patch descriptors from the target reference only",
        "insufficient_descriptor_variation": mean_variance <= 1e-12,
    }


class CenteredScratchMatcher:
    """CNN-only region matcher after frozen support-only centering and scaling."""
    name = "centered_scratch_regions"

    def __init__(self, base_matcher, *, shrinkage=.5, mode="diagonal",
                 slice_penalty=.92, variance_floor=1e-6, cache_size=64):
        if mode not in {"diagonal", "zca"}:
            raise ValueError("mode must be diagonal or zca")
        if not math.isfinite(shrinkage) or not 0 < shrinkage <= 1:
            raise ValueError("shrinkage must be finite and in (0,1]")
        if not math.isfinite(slice_penalty) or not 0 < slice_penalty <= 1:
            raise ValueError("slice_penalty must be finite and in (0,1]")
        if not math.isfinite(variance_floor) or variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.base_matcher = base_matcher
        self.shrinkage, self.mode = float(shrinkage), str(mode)
        self.slice_penalty, self.variance_floor = float(slice_penalty), float(variance_floor)
        self.cache_size = cache_size
        self.references = OrderedDict()
        self._assert_scratch(base_matcher)

    @staticmethod
    def _assert_scratch(matcher):
        provenance = matcher.provenance()
        if provenance.get("external_training_images") != 0 or provenance.get("external_pretraining", False) not in (False, None):
            raise ValueError("Centered matcher requires a verified zero-external-training-image CNN")

    def _base(self, reference):
        if hasattr(self.base_matcher, "_features"):
            return self.base_matcher
        if hasattr(self.base_matcher, "_key") and hasattr(self.base_matcher, "matchers"):
            return self.base_matcher.matchers[self.base_matcher._key(reference)]
        raise TypeError("CNN must provide bound patch features or a reference router")

    def _key(self, reference, base):
        return image_digest(reference), getattr(base, "checkpoint_sha256", id(base))

    def prepare(self, reference):
        reference = rgb_uint8(reference)
        if hasattr(self.base_matcher, "prepare"):
            self.base_matcher.prepare(reference)
        base = self._base(reference)
        self._assert_scratch(base)
        if base.metadata.get("reference_rgb_sha256") != image_digest(reference):
            raise ValueError("Checkpoint was trained on a different reference image")
        if base.metadata.get("insufficient_informative_patches", False):
            return
        key = self._key(reference, base)
        if key not in self.references:
            features, points = base._features(reference)
            if self.mode == "diagonal":
                mean, transform, diagnostics = fit_reference_diagonal(features, self.shrinkage, self.variance_floor)
            else:
                mean, transform, diagnostics = fit_reference_whitener(features, self.shrinkage, self.variance_floor)
                diagnostics = {**diagnostics, "mode": "zca",
                               "insufficient_descriptor_variation": diagnostics["mean_variance"] <= 1e-12}
            normalized = apply_reference_whitener(features, mean, transform).astype(np.float32)
            points = np.array(points, np.float64, copy=True)
            if points.shape != (len(normalized), 2) or not np.isfinite(points).all():
                raise ValueError("CNN must provide one finite point per reference descriptor")
            for array in (normalized, points, mean, transform):
                array.setflags(write=False)
            self.references[key] = (normalized, points, mean, transform, diagnostics)
            while len(self.references) > self.cache_size:
                self.references.popitem(last=False)
        self.references.move_to_end(key)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        self.prepare(reference)
        base = self._base(reference)
        evidence = {
            "method": self.name, "cnn_method": base.name,
            "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "query_weight_updates": 0,
            "normalization_fit_images": 1, "normalization_fit_query_images": 0,
            "score_is_probability": False, "geometry_verified": False,
            "box_is_search_proposal": True, "handcrafted_image_scores": False,
        }
        if base.metadata.get("insufficient_informative_patches", False):
            return MatchResult(0., {**evidence, "reason": "insufficient informative reference patches; scratch CNN abstained"})
        a, pa, mean, transform, diagnostics = self.references[self._key(reference, base)]
        evidence["normalization"] = dict(diagnostics)
        if diagnostics["insufficient_descriptor_variation"]:
            return MatchResult(0., {**evidence, "reason": "insufficient variation in reference CNN descriptors"})
        features, pb = base._features(query)
        b = apply_reference_whitener(features, mean, transform).astype(np.float32)
        rb, qb = region_boxes(reference.shape, reference=True), region_boxes(query.shape)
        scores, details = _cnn_region_evidence(a, b, pa, pb, rb, qb)
        penalty = np.full(len(rb), self.slice_penalty, np.float32)
        penalty[0] = 1.
        scores *= penalty[None, :]
        qi, ri = np.unravel_index(int(scores.argmax()), scores.shape)
        detail = details.get((qi, ri), {})
        evidence.update({
            "reference_slices": len(rb), "query_windows": len(qb),
            "reference_slice_xyxy": list(rb[ri]), "reference_slice_penalty": float(penalty[ri]),
            "cnn_components": {key: value for key, value in detail.items() if not key.endswith("_ids")},
            "all_regions_share_full_image_cnn_features": True,
        })
        ref_ids, query_ids = detail.get("reference_ids", []), detail.get("query_ids", [])
        return MatchResult(
            float(scores[qi, ri]), evidence,
            points_reference=np.asarray(pa)[ref_ids].astype(float).tolist(),
            points_query=np.asarray(pb)[query_ids].astype(float).tolist(),
            bbox_xyxy=list(map(float, qb[qi])) if scores[qi, ri] > 0 else None,
        )

    def provenance(self):
        return {
            "method": self.name, "external_pretraining": False, "external_training_images": 0,
            "target_reference_images": 1, "query_weight_updates": 0,
            "handcrafted_image_scores": False, "score_is_probability": False,
            "normalization": "frozen reference mean subtraction and shrunk " + self.mode + " variance normalization; per-descriptor L2",
            "normalization_fit": "ONE reference image's CNN patch descriptors only; no queries or labels",
            "normalization_fit_query_images": 0, "checkpoint_retrained": False,
            "mode": self.mode, "shrinkage": self.shrinkage,
            "variance_floor": self.variance_floor, "slice_penalty": self.slice_penalty,
            "query_windows_max": 30, "reference_slices_max": 6,
            "cnn_region_score": "half centered mean-descriptor cosine and half centered partial reciprocal patch support",
            "cnn": self.base_matcher.provenance(),
            "reference_fits": [{"reference_sha256": key[0], **dict(state[4])} for key, state in self.references.items()],
            "limitation": "Reference centering removes common feature offsets but can remove useful shared category information. Scores still need development calibration; one-image covariance is not a category distribution.",
        }

