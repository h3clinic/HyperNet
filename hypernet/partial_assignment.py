"""Exact partial *linear* assignment for small, untrained part descriptors.

The unmatched-node construction follows the optimal-partial-transport idea in
Ratnayaka et al., Learning Partial Graph Matching via Optimal Partial Transport,
ICLR 2025: https://arxiv.org/abs/2410.16718 . This module is an engineering
adaptation of the assignment objective, not a reproduction of that paper's
trained graph matcher, and not a Gromov--Wasserstein solver.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def partial_assignment(
    similarity: np.ndarray,
    row_weights: np.ndarray | None = None,
    min_similarity: float = 0.2,
    min_coverage: float = 0.6,
    geometry_penalty: np.ndarray | float | None = None,
) -> dict:
    """Match each support/query part at most once, permitting unmatched parts.

    Maximize sum_(i,j in M) w_i * (s_ij - min_similarity - p_ij), where
    normalized support weights sum to one. Each row and column has capacity
    one; a zero-gain dummy node provides the unmatched choice. Only strictly
    positive real matches are reported. Nonfinite similarities or penalties
    forbid an edge. Finite similarities are clipped to [-1, 1] to absorb
    numerical cosine overshoot; geometry penalties must be nonnegative.

    Coverage is matched support weight, and the score is total positive gain
    divided by (1 - min_similarity). It is NEVER renormalized by matched mass.
    Coverage below min_coverage gates the returned score to zero, but retains
    the assignment and ungated_score for diagnostics. This is a post-solve
    gate, not a constrained minimum-coverage optimization. All-zero support
    weights represent no evidence and return an empty, zero-score result.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2:
        raise ValueError("similarity must be a two-dimensional matrix")
    if not np.isfinite(min_similarity) or not -1.0 <= min_similarity < 1.0:
        raise ValueError("min_similarity must be finite and in [-1, 1)")
    if not np.isfinite(min_coverage) or not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be finite and in [0, 1]")
    n_rows, n_cols = similarity.shape
    if row_weights is None:
        weights = np.ones(n_rows, dtype=np.float64)
    else:
        weights = np.asarray(row_weights, dtype=np.float64)
        if weights.shape != (n_rows,):
            raise ValueError("row_weights must have one entry per support row")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("row_weights must be finite and nonnegative")
    # Rescale before summing so otherwise valid, large weights cannot overflow.
    weight_max = float(np.max(weights)) if n_rows else 0.0
    weights = weights.copy()
    if weight_max > 0:
        weights /= weight_max
        weights /= weights.sum()

    if geometry_penalty is None:
        penalty = np.zeros_like(similarity)
    else:
        try:
            penalty = np.broadcast_to(np.asarray(geometry_penalty, dtype=np.float64), similarity.shape)
        except ValueError as exc:
            raise ValueError("geometry_penalty must broadcast to similarity.shape") from exc
        if np.any(penalty[np.isfinite(penalty)] < 0):
            raise ValueError("finite geometry penalties must be nonnegative")

    result = {
        "score": 0.0, "ungated_score": 0.0, "coverage": 0.0,
        "coverage_passed": False, "rows": [], "cols": [], "gains": [],
        "assigned_scores": [], "matched_similarities": [],
        "matched_row_weights": [], "total_gain": 0.0,
    }
    if not n_rows or not n_cols or weight_max == 0:
        return result
    valid = np.isfinite(similarity) & np.isfinite(penalty)
    clipped = np.clip(np.where(valid, similarity, 0.0), -1.0, 1.0)
    safe_penalty = np.where(valid, penalty, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        gains = (clipped - min_similarity - safe_penalty) * weights[:, None]
    # Invalid/nonpositive edges can never improve on a zero-gain dummy. Use
    # finite costs for SciPy even if a supplied finite penalty overflowed.
    gains = np.where(valid & np.isfinite(gains), gains, -1.0)
    augmented = np.zeros((n_rows + n_cols, n_rows + n_cols), dtype=np.float64)
    augmented[:n_rows, :n_cols] = gains
    all_rows, all_cols = linear_sum_assignment(augmented, maximize=True)
    real = (all_rows < n_rows) & (all_cols < n_cols)
    rows, cols = all_rows[real], all_cols[real]
    positive = gains[rows, cols] > 0.0
    rows, cols = rows[positive], cols[positive]
    selected_gains = gains[rows, cols]
    coverage = float(np.clip(weights[rows].sum(), 0.0, 1.0))
    total_gain = float(selected_gains.sum())
    ungated = float(np.clip(total_gain / (1.0 - min_similarity), 0.0, 1.0))
    passed = bool(rows.size and coverage + 1e-12 >= min_coverage)
    result.update(
        score=ungated if passed else 0.0, ungated_score=ungated,
        coverage=coverage, coverage_passed=passed,
        rows=rows.tolist(), cols=cols.tolist(), gains=selected_gains.tolist(),
        assigned_scores=(clipped[rows, cols] - safe_penalty[rows, cols]).tolist(),
        matched_similarities=clipped[rows, cols].tolist(),
        matched_row_weights=weights[rows].tolist(), total_gain=total_gain,
    )
    return result
