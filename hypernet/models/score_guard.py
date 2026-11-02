"""Preserve analytic matching, then bound the CNN's scalar-score contribution.

Feature fusion can change reciprocal argmax correspondences discontinuously.
This adapter completes the two matchers separately and blends their bounded
scores, leaving the analytic correspondences untouched. Its bound concerns raw
scores, never calibrated decisions, detection rates, or category accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import math
import numpy as np

from ..types import rgb_uint8
from .analytic_patches import AnalyticPatchMatcher
from .cnn import image_digest


def bounded_score_mix(analytic_score, learned_score, learned_weight=.2):
    """For component scores in [0,1], |mixture - analytic| <= learned_weight."""
    if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
        raise ValueError("learned_weight must be finite and in [0,1]")
    for name, score in (("analytic", analytic_score), ("learned", learned_score)):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"{name} score must be finite and in [0,1]")
    if learned_weight == 0:
        return float(analytic_score)
    if learned_weight == 1:
        return float(learned_score)
    return float((1 - learned_weight) * analytic_score + learned_weight * learned_score)


def _provenance_nodes(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _provenance_nodes(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _provenance_nodes(item)


def _assert_scratch(matcher):
    provenance = matcher.provenance()
    if provenance.get("external_training_images") != 0:
        raise ValueError("Score guard requires an explicit zero-external-training model")
    for node in _provenance_nodes(provenance):
        if node.get("external_pretraining", False) not in (False, None):
            raise ValueError("Score guard rejects pretrained weights, including contradictory nested provenance")
        if "external_training_images" in node and node["external_training_images"] != 0:
            raise ValueError("Score guard rejects external training, including nested provenance")
        for key in ("target_reference_images", "target_reference_images_per_model"):
            if key in node and node[key] != 1:
                raise ValueError("Score guard requires exactly one reference per model")
    return provenance


class AnalyticScoreGuard:
    name = "analytic_score_guard_v4"

    def __init__(self, base_matcher, *, analytic_matcher=None, learned_weight=.2):
        if not math.isfinite(learned_weight) or not 0 <= learned_weight <= 1:
            raise ValueError("learned_weight must be finite and in [0,1]")
        _assert_scratch(base_matcher)
        self.base_matcher = base_matcher
        self.analytic_matcher = analytic_matcher if analytic_matcher is not None else AnalyticPatchMatcher()
        fixed = _assert_scratch(self.analytic_matcher)
        if fixed.get("learned_parameters") != 0 or fixed.get("target_weight_updates") != 0:
            raise ValueError("The analytic component must be explicitly fixed and untrained")
        self.learned_weight = float(learned_weight)
        self._learned_views = {}

    def _bound(self, reference):
        if hasattr(self.base_matcher, "metadata"):
            return self.base_matcher
        if hasattr(self.base_matcher, "matchers") and hasattr(self.base_matcher, "_key"):
            return self.base_matcher.matchers[self.base_matcher._key(reference)]
        raise TypeError("Score guard needs a reference-bound matcher or a reference router")

    def _learned_only(self, base):
        if not hasattr(base, "with_learned_weight"):
            return base
        # GuardedShapeMatcher variants share exact weights and raw feature cache.
        # Never accidentally score a feature mixture as the CNN-only component.
        key = (id(base), getattr(base, "checkpoint_sha256", None))
        if key not in self._learned_views:
            self._learned_views[key] = base.with_learned_weight(1.)
        return self._learned_views[key]

    def prepare(self, reference):
        reference = rgb_uint8(reference)
        _assert_scratch(self.base_matcher)
        if hasattr(self.base_matcher, "prepare"):
            self.base_matcher.prepare(reference)
        base = self._bound(reference)
        _assert_scratch(base)
        metadata = base.metadata
        if metadata.get("reference_rgb_sha256") != image_digest(reference):
            raise ValueError("Checkpoint was trained on a different reference image")
        if metadata.get("external_training_images") != 0 or metadata.get("external_pretraining") is not False:
            raise ValueError("Bound CNN metadata must establish zero external training")
        if metadata.get("target_reference_images") != 1:
            raise ValueError("Bound CNN must be trained on exactly one reference")
        if hasattr(self.analytic_matcher, "prepare"):
            self.analytic_matcher.prepare(reference)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        self.prepare(reference)
        base = self._bound(reference)
        analytic = self.analytic_matcher.predict(reference, query)
        analytic_score = float(analytic.score)
        if not math.isfinite(analytic_score) or not 0 <= analytic_score <= 1:
            raise ValueError("Analytic score must be finite and in [0,1]")
        weight, learned, reason = self.learned_weight, None, None
        if not np.any(reference != reference[0, 0]) or analytic.evidence.get("reason") == "spatially_uniform_reference":
            # A numerical CNN response cannot revive an uninformative reference.
            weight, reason = 0., "spatially_uniform_reference"
            if analytic_score != 0:
                raise ValueError("An uninformative analytic reference must abstain with score zero")
        elif base.metadata.get("insufficient_informative_patches"):
            weight, reason = 0., "insufficient_cnn_reference_information"
        elif weight == 0:
            reason = "analytic_only_weight"
        else:
            learned = self._learned_only(base).predict(reference, query)
        score = (analytic_score if weight == 0 else
                 bounded_score_mix(analytic_score, float(learned.score), weight))
        result = deepcopy(analytic)
        result.score = score
        result.evidence = {
            **deepcopy(analytic.evidence), "method": self.name,
            "external_training_images": 0, "external_pretraining": False,
            "target_reference_images": 1, "query_weight_updates": 0,
            "learned_weight": self.learned_weight, "effective_learned_weight": weight,
            "analytic_score": analytic_score,
            "learned_score": float(learned.score) if learned is not None else None,
            "analytic_evidence": deepcopy(analytic.evidence),
            "learned_evidence": deepcopy(learned.evidence) if learned is not None else None,
            "absolute_change_from_analytic_score": abs(score - analytic_score),
            "maximum_raw_score_change": weight,
            "bound_scope": "Raw scalar score only; calibrated decisions and recall are not protected",
            "analytic_correspondences_unchanged": True,
            "correspondence_source": "fixed analytic matcher; these points do not explain the CNN score",
            "handcrafted_cues_not_learned": True,
            "geometry_verified": False, "score_is_probability": False,
        }
        if reason is not None:
            result.evidence["score_guard_fallback"] = reason
        return result

    def provenance(self):
        return {
            "method": self.name, "external_training_images": 0, "external_pretraining": False,
            "target_reference_images": 1, "query_weight_updates": 0,
            "learned_weight": self.learned_weight, "analytic_weight": 1 - self.learned_weight,
            "fusion": "(1-alpha)*unchanged analytic score + alpha*separately completed learned-only CNN score",
            "raw_score_perturbation_bound": self.learned_weight,
            "bound_scope": "Both component scores must lie in[0,1]; no category-accuracy or calibrated-threshold guarantee",
            "analytic_correspondences_unchanged": True,
            "handcrafted_cues_not_learned": True,
            "cnn_score_uses_learned_only": True, "score_is_probability": False,
            "cnn": self.base_matcher.provenance(), "fixed_representation": self.analytic_matcher.provenance(),
            "fallback": "Uniform reference abstains; insufficient CNN training preserves the exact analytic score",
            "limitation": "Preserving the analytic score path limits numeric change but can still alter rankings and threshold decisions. The CNN can add no new category observations.",
        }
