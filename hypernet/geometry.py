"""Small, guarded geometry helpers for reference-to-query image matching.

These helpers model a planar object (or a pure camera rotation). They do not
recover unseen surfaces or solve general non-rigid / 3-D correspondence.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class GeometryResult:
    homography: np.ndarray | None
    inliers: np.ndarray
    projected_corners: np.ndarray | None
    bbox_xyxy: list[float] | None
    median_error: float | None
    valid: bool
    reason: str


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Classical matching requires opencv-python-headless") from exc
    return cv2


def project_points(points, homography):
    points = np.asarray(points, dtype=np.float64)
    h = np.asarray(homography, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ h.T
    denominator = homogeneous[:, 2]
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(denominator) > 1e-9)
    result[valid] = homogeneous[valid, :2] / denominator[valid, None]
    return result


def _area(points):
    points = np.asarray(points)
    return float(abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                     np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2)


def _noncollinear(points, tolerance=1e-3):
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return len(singular) == 2 and singular[0] > 1e-6 and singular[1] / singular[0] > tolerance


def estimate_homography(points_reference, points_query, reference_shape, query_shape,
                        reprojection_threshold=4.0, min_inliers=6, seed=0):
    """Fit H mapping reference pixels to query pixels, with conservative checks.

    ``reference_shape`` and ``query_shape`` are (height, width[, channels]).
    The RANSAC threshold is in pixels at the input resolution. Projected corners
    must stay finite, convex and on the same side of the projective horizon.
    A partly off-screen target is accepted; extreme/unbounded fits are rejected.
    An invalid estimate has no H, no bounding box, and an all-False inlier mask.
    """
    cv2 = _cv2()
    p0 = np.asarray(points_reference, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(points_query, dtype=np.float64).reshape(-1, 2)
    if len(p0) != len(p1):
        raise ValueError("Correspondence arrays must have equal length")
    rejected = lambda reason: GeometryResult(None, np.zeros(len(p0), dtype=bool),
                                             None, None, None, False, reason)
    if len(p0) < max(4, min_inliers):
        return rejected("too_few_correspondences")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        return rejected("nonfinite_correspondences")
    if len(np.unique(p0, axis=0)) < 4 or len(np.unique(p1, axis=0)) < 4:
        return rejected("duplicate_correspondences")
    if not _noncollinear(p0) or not _noncollinear(p1):
        return rejected("collinear_correspondences")
    # OpenCV RANSAC uses a process-global RNG. The runner executes sequentially.
    cv2.setRNGSeed(int(seed) & 0x7FFFFFFF)
    try:
        h, mask = cv2.findHomography(p0, p1, cv2.RANSAC,
                                    float(reprojection_threshold), maxIters=3000,
                                    confidence=0.995)
    except cv2.error:
        return rejected("homography_fit_failed")
    if h is None or mask is None or not np.isfinite(h).all():
        return rejected("homography_fit_failed")
    if abs(h[2, 2]) < 1e-12:
        return rejected("unstable_normalization")
    h = h / h[2, 2]
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < min_inliers:
        return rejected("too_few_inliers")
    if not _noncollinear(p0[inliers]) or not _noncollinear(p1[inliers]):
        return rejected("collinear_inliers")
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return rejected("singular_homography")
    if not np.isfinite(inverse).all():
        return rejected("singular_homography")
    rh, rw = reference_shape[:2]
    qh, qw = query_shape[:2]
    corners = np.array([[0, 0], [rw - 1, 0], [rw - 1, rh - 1], [0, rh - 1]], dtype=float)
    denominator = np.column_stack((corners, np.ones(4))) @ h[2]
    if np.any(np.abs(denominator) < 1e-6) or not (np.all(denominator > 0) or np.all(denominator < 0)):
        return rejected("projective_horizon_crosses_reference")
    projected = project_points(corners, h)
    if not np.isfinite(projected).all():
        return rejected("nonfinite_projection")
    contour = projected.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        return rejected("nonconvex_projection")
    area = _area(projected)
    query_area = float(qh * qw)
    if area < 16 or area > 16 * query_area:
        return rejected("implausible_projected_area")
    if np.any(np.abs(projected) > 8 * max(qh, qw, rh, rw)):
        return rejected("extreme_projection")
    x0, y0 = np.maximum(projected.min(axis=0), [0, 0])
    x1, y1 = np.minimum(projected.max(axis=0), [qw, qh])
    if x1 <= x0 or y1 <= y0:
        return rejected("no_query_overlap")
    query_corners = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh]], dtype=np.float32)
    overlap_area, _ = cv2.intersectConvexConvex(projected.astype(np.float32), query_corners)
    if not np.isfinite(overlap_area) or overlap_area < 4:
        return rejected("no_query_overlap")
    # Recompute residuals rather than trusting an implementation-dependent mask.
    errors = np.linalg.norm(project_points(p0, h) - p1, axis=1)
    inliers = inliers & np.isfinite(errors) & (errors <= reprojection_threshold * 1.5)
    if inliers.sum() < min_inliers:
        return rejected("too_few_verified_inliers")
    return GeometryResult(h, inliers, projected,
                          [float(x0), float(y0), float(x1), float(y1)],
                          float(np.median(errors[inliers])), True, "ok")


def spatial_coverage(points, image_shape, grid=4):
    """Fraction of grid cells touched; sparse/distributed evidence beats a clump.

    This is intentionally not claimed to estimate visible object fraction.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return 0.0
    height, width = image_shape[:2]
    cells = np.floor(points / [max(width, 1), max(height, 1)] * grid)
    cells = np.clip(cells.astype(int), 0, grid - 1)
    return float(len(np.unique(cells, axis=0)) / (grid * grid))


def graph_consistency(points_reference, points_query, neighbors=3):
    """Mean k-nearest-neighbor edge agreement in two correspondence point sets.

    Rotation and uniform scaling preserve these neighborhoods; perspective and
    non-rigid changes need not. This is an auxiliary heuristic, correlated with
    homography evidence, and is explicitly ablatable. It is not a learned graph
    network or a proof of object identity.
    """
    a = np.asarray(points_reference, dtype=float).reshape(-1, 2)
    b = np.asarray(points_query, dtype=float).reshape(-1, 2)
    if len(a) != len(b):
        raise ValueError("Correspondence arrays must have equal length")
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0
    # At most 256 deterministic samples avoids quadratic growth for dense matches.
    if len(a) > 256:
        indices = np.linspace(0, len(a) - 1, 256, dtype=int)
        a, b = a[indices], b[indices]
    k = min(int(neighbors), len(a) - 1)
    def nearest(points):
        distances = np.sum((points[:, None] - points[None]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        return np.argsort(distances, axis=1, kind="stable")[:, :k]
    na, nb = nearest(a), nearest(b)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(na, nb)]))
