"""Training-free local feature matching baselines; OpenCV supplies SIFT/ORB."""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import numpy as np
from ..types import MatchResult, rgb_uint8
from ..geometry import _cv2, estimate_homography, spatial_coverage, graph_consistency


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray | None


class LocalFeatureMatcher:
    """Shared extractor, mutual ratio matching and geometry, without learning.

    LRU caches contain only image features and are keyed by image bytes, image
    shape, and the extractor configuration. Test labels never enter the model.
    """
    def __init__(self, detector="sift", max_features=1200, ratio=0.78,
                 reprojection_threshold=4.0, min_inliers=6, cache_size=128):
        cv2 = _cv2()
        if not 0 < ratio < 1:
            raise ValueError("ratio must lie strictly between 0 and 1")
        if max_features < 16 or min_inliers < 4:
            raise ValueError("max_features must be >=16 and min_inliers >=4")
        if reprojection_threshold <= 0 or cache_size < 0:
            raise ValueError("positive reprojection_threshold and nonnegative cache_size required")
        self.detector_kind = detector
        self.max_features = int(max_features)
        self.ratio = float(ratio)
        self.reprojection_threshold = float(reprojection_threshold)
        self.min_inliers = int(min_inliers)
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        if detector == "sift":
            self._detector = cv2.SIFT_create(nfeatures=self.max_features, contrastThreshold=0.025)
            self._norm = cv2.NORM_L2
        elif detector == "orb":
            self._detector = cv2.ORB_create(nfeatures=self.max_features, fastThreshold=10)
            self._norm = cv2.NORM_HAMMING
        else:
            raise ValueError("detector must be 'sift' or 'orb'")
        self.name = detector + "_ransac"

    def _features(self, image):
        cv2 = _cv2()
        image = rgb_uint8(image)
        signature = (self.detector_kind, self.max_features, image.shape,
                     hashlib.sha256(image.tobytes()).digest())
        if signature in self._cache:
            self._cache.move_to_end(signature)
            return self._cache[signature]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)
        features = Features(np.array([p.pt for p in keypoints], dtype=np.float64).reshape(-1, 2), descriptors)
        if self.cache_size:
            self._cache[signature] = features
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return features

    def correspondences(self, reference, query):
        cv2 = _cv2()
        f0, f1 = self._features(reference), self._features(query)
        empty = (np.empty((0, 2)), np.empty((0, 2)), np.empty(0), len(f0.points), len(f1.points))
        if f0.descriptors is None or f1.descriptors is None or min(len(f0.points), len(f1.points)) < 2:
            return empty
        matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        forwards = matcher.knnMatch(f0.descriptors, f1.descriptors, k=2)
        backwards = matcher.knnMatch(f1.descriptors, f0.descriptors, k=2)
        def passes(pairs):
            result = {}
            for pair in pairs:
                if len(pair) != 2:
                    continue
                first, second = pair
                if second.distance > 1e-12 and first.distance < self.ratio * second.distance:
                    result[first.queryIdx] = (first.trainIdx, first.distance / second.distance)
            return result
        forward, backward = passes(forwards), passes(backwards)
        accepted = [(i, j, ratio) for i, (j, ratio) in forward.items()
                    if j in backward and backward[j][0] == i]
        if not accepted:
            return empty
        # Sorting yields reproducible point ordering (including graph sampling).
        accepted.sort(key=lambda x: (x[0], x[1]))
        i0, i1, ratios = zip(*accepted)
        return f0.points[list(i0)], f1.points[list(i1)], np.asarray(ratios), len(f0.points), len(f1.points)

    def predict(self, reference, query):
        reference, query = rgb_uint8(reference), rgb_uint8(query)
        p0, p1, ratios, n0, n1 = self.correspondences(reference, query)
        geometry = estimate_homography(p0, p1, reference.shape, query.shape,
                                       self.reprojection_threshold, self.min_inliers)
        n = int(geometry.inliers.sum())
        fraction = n / max(len(p0), 1)
        # Saturation makes six accidental inliers less persuasive than broad support.
        support = float(1 - np.exp(-n / 18.0))
        score = support * fraction if geometry.valid else 0.0
        evidence = {"reference_features": n0, "query_features": n1,
                    "mutual_ratio_matches": len(p0), "inlier_count": n,
                    "inlier_fraction": fraction, "support": support,
                    "geometry_valid": geometry.valid, "geometry_reason": geometry.reason,
                    "median_reprojection_error_px": geometry.median_error,
                    "score_is_probability": False}
        return MatchResult(float(score), evidence, p0.tolist(), p1.tolist(),
                           geometry.inliers.tolist(),
                           geometry.homography.tolist() if geometry.valid else None,
                           geometry.bbox_xyxy)

    def provenance(self):
        return {"model": self.name, "implementation": "OpenCV " + self.detector_kind.upper() + " + mutual ratio test + RANSAC",
                "external_training_images": 0, "reference_images_per_target": 1,
                "learned_parameters": False, "adaptation": "extract and cache reference features; no optimization",
                "parameters": {"max_features": self.max_features, "ratio": self.ratio,
                               "reprojection_threshold_px": self.reprojection_threshold,
                               "min_inliers": self.min_inliers},
                "scope": "planar/local appearance correspondence; not a general semantic category classifier",
                "sources": ["https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf",
                            "https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html"]}


class ORBMatcher(LocalFeatureMatcher):
    def __init__(self, max_features=1200, ratio=0.78, **kwargs):
        super().__init__("orb", max_features, ratio, **kwargs)


class SIFTMatcher(LocalFeatureMatcher):
    def __init__(self, max_features=1200, ratio=0.78, **kwargs):
        super().__init__("sift", max_features, ratio, **kwargs)
