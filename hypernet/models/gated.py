"""Geometry-gated late fusion: a second exploratory engineering iteration.

The CNN may modulate valid geometric evidence but cannot create a match when
geometric verification failed. This is an explicit heuristic, not a new trained
architecture or a reproduction of any cited paper.
"""
from __future__ import annotations
import math

from ..types import MatchResult


class GatedHybridMatcher:
    name = "hypernet_hybrid_gated"

    def __init__(self, geometry_model, cnn_model, cnn_weight: float = .25):
        if not math.isfinite(cnn_weight) or not 0 <= cnn_weight <= 1:
            raise ValueError("cnn_weight must be finite and between zero and one")
        self.geometry_model = geometry_model
        self.cnn_model = cnn_model
        self.cnn_weight = float(cnn_weight)

    def predict(self, reference, query):
        geom = self.geometry_model.predict(reference, query)
        geometry_score = float(geom.score)
        if not math.isfinite(geometry_score):
            raise ValueError("Geometry component returned a non-finite score")
        # The strict flag deliberately excludes missing or disabled geometric
        # verification, even when a descriptor-only ablation has a high score.
        geometry_valid = geom.evidence.get("geometry_valid") is True
        geometry_score_clipped = min(1., max(0., geometry_score))
        gate_passed = geometry_valid and geometry_score_clipped > 0.
        cnn_score, cnn_evidence, multiplier = None, {"skipped": True}, 0.
        if gate_passed:
            cnn = self.cnn_model.predict(reference, query)
            cnn_score = float(cnn.score)
            if not math.isfinite(cnn_score):
                raise ValueError("CNN component returned a non-finite score")
            clipped_cnn_score = min(1., max(0., cnn_score))
            multiplier = (1. - self.cnn_weight) + self.cnn_weight * clipped_cnn_score
            cnn_evidence = cnn.evidence
        score = geometry_score_clipped * multiplier if gate_passed else 0.
        return MatchResult(
            score=float(score),
            evidence={
                "geometry_score": geometry_score,
                "geometry_score_clipped": geometry_score_clipped,
                "geometry_valid": geometry_valid,
                "geometry_gate_passed": gate_passed,
                "cnn_score": cnn_score,
                "cnn_weight": self.cnn_weight,
                "cnn_multiplier": multiplier,
                "cnn_skipped": not gate_passed,
                "geometry": geom.evidence,
                "cnn": cnn_evidence,
                "score_is_probability": False,
                "stage": "second exploratory iteration",
            },
            points_reference=geom.points_reference,
            points_query=geom.points_query,
            inliers=geom.inliers,
            homography=geom.homography,
            bbox_xyxy=geom.bbox_xyxy,
        )

    def provenance(self):
        return {
            "method": self.name,
            "stage": "second exploratory iteration after initial pilot inspection",
            "formula": "clip(geometry,0,1) * ((1-cnn_weight) + cnn_weight*clip(cnn,0,1)) when geometry_valid=True; otherwise 0",
            "cnn_weight": self.cnn_weight,
            "weight_selection": "Fixed engineering revision; not learned or numerically fitted. Evaluate on fresh identities after the initial pilot.",
            "geometry_gate": "Requires explicit geometry_valid=True and positive geometry score",
            "target_reference_images": 1,
            "target_weight_updates": 0,
            "components": [self.geometry_model.provenance(), self.cnn_model.provenance()],
            "external_pretraining": "Inherited from CNN component; not zero-data training",
            "paper_reproduction": False,
            "novelty": "Not established. A conservative combination of existing matching mechanisms.",
            "limitations": [
                "Cannot recover true matches rejected by the geometric branch.",
                "A valid homography may still fit a wrong repeated or planar pattern.",
                "Reducing scores does not guarantee lower false-positive rates after independently recalibrating thresholds.",
                "Initial pilot results informed this revision and must remain exploratory evidence.",
            ],
        }
