"""Experimental reference-part evidence fusion, with explicit ablation switches.

This is an original starter heuristic inspired by published ideas. It is not a
reimplementation of OS2D, Slime, DTM, a GNN, or a trained CNN.
"""
import numpy as np
from .classical import LocalFeatureMatcher
from ..types import MatchResult, rgb_uint8
from ..geometry import estimate_homography, spatial_coverage, graph_consistency


class PartGraphMatcher(LocalFeatureMatcher):
    def __init__(self, max_features=1200, ratio=0.78, geometry=True,
                 coverage=True, graph=True, **kwargs):
        super().__init__("sift", max_features, ratio, **kwargs)
        self.use_geometry = bool(geometry)
        self.use_coverage = bool(coverage)
        self.use_graph = bool(graph)
        suffix = "".join("_no_" + name for name, enabled in
                         [("geometry", geometry), ("coverage", coverage), ("graph", graph)] if not enabled)
        self.name = "partgraph" + suffix

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        p0, p1, ratios, n0, n1 = self.correspondences(reference, query)
        geometry = None
        if self.use_geometry:
            geometry = estimate_homography(p0, p1, reference.shape, query.shape,
                                           self.reprojection_threshold, self.min_inliers)
            selected = geometry.inliers
        else:
            selected = np.ones(len(p0), dtype=bool)
        n = int(selected.sum())
        fraction = n / max(len(p0), 1)
        support = float(1 - np.exp(-n / 18.0))
        # Ratio 0 indicates maximal distinctiveness; ratio near the gate indicates
        # weak descriptor separation. Fixed parameters are not fitted on test data.
        distinctiveness = float(np.mean(1 - ratios[selected])) if n else 0.0
        coverage = spatial_coverage(p0[selected], reference.shape)
        graph = graph_consistency(p0[selected], p1[selected]) if n else 0.0
        terms = {"support": support,
                 "geometry_fraction": fraction if self.use_geometry else 1.0,
                 "distinctiveness_factor": 0.5 + 0.5 * distinctiveness,
                 "coverage_factor": 0.5 + 0.5 * np.sqrt(coverage) if self.use_coverage else 1.0,
                 "graph_factor": 0.5 + 0.5 * graph if self.use_graph else 1.0}
        score = float(np.prod(list(terms.values()))) if n >= self.min_inliers else 0.0
        evidence = {"reference_features": n0, "query_features": n1,
                    "mutual_ratio_matches": len(p0), "inlier_count": n,
                    "inlier_fraction": fraction, "support": support,
                    "descriptor_distinctiveness": distinctiveness,
                    "reference_grid_coverage": coverage, "graph_consistency": graph,
                    "geometry_valid": geometry.valid if geometry else None,
                    "geometry_reason": geometry.reason if geometry else "disabled_for_ablation",
                    "median_reprojection_error_px": geometry.median_error if geometry else None,
                    "score_terms": {k: float(v) for k, v in terms.items()},
                    "ablation": {"geometry": self.use_geometry, "coverage": self.use_coverage, "graph": self.use_graph},
                    "score_is_probability": False}
        return MatchResult(score, evidence, p0.tolist(), p1.tolist(), selected.tolist(),
                           geometry.homography.tolist() if geometry and geometry.valid else None,
                           geometry.bbox_xyxy if geometry else None)

    def provenance(self):
        result = super().provenance()
        result.update({"implementation": "experimental SIFT part-evidence and neighborhood consistency fusion",
                       "ablation": {"geometry": self.use_geometry, "coverage": self.use_coverage, "graph": self.use_graph},
                       "novelty_status": "unestablished; exploratory combination of known mechanisms",
                       "paper_reproduction": False,
                       "sources": ["https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf",
                                   "https://arxiv.org/abs/2003.06800",
                                   "https://arxiv.org/abs/2305.08946",
                                   "https://arxiv.org/abs/1604.03518"]})
        return result
