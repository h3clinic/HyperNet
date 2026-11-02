"""Fixed, auditable late fusion. This is not an end-to-end trained architecture."""
from hypernet.types import MatchResult

class HybridMatcher:
    name = "hypernet_hybrid"
    def __init__(self, geometry_model, cnn_model, geometry_weight=0.65):
        if not 0 <= geometry_weight <= 1:
            raise ValueError("geometry_weight must be between zero and one")
        self.geometry_model = geometry_model
        self.cnn_model = cnn_model
        self.geometry_weight = geometry_weight

    def predict(self, reference, query):
        geom = self.geometry_model.predict(reference, query)
        cnn = self.cnn_model.predict(reference, query)
        w = self.geometry_weight
        return MatchResult(
            score=float(w * geom.score + (1-w) * cnn.score),
            evidence={"geometry_score": geom.score, "cnn_score": cnn.score,
                      "geometry_weight": w, "geometry": geom.evidence, "cnn": cnn.evidence,
                      "score_is_probability": False},
            points_reference=geom.points_reference, points_query=geom.points_query,
            inliers=geom.inliers, homography=geom.homography, bbox_xyxy=geom.bbox_xyxy)

    def provenance(self):
        return {"method": self.name, "stage": "experimental fixed late fusion",
                "geometry_weight": self.geometry_weight,
                "weight_selection": "fixed engineering default, not optimized on test results",
                "target_reference_images": 1,
                "components": [self.geometry_model.provenance(), self.cnn_model.provenance()],
                "external_pretraining": "Inherited from CNN component; not zero-data training",
                "novelty": "not established; combination is an experimental starter"}
