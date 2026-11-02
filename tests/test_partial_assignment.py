"""Independent small matching oracle and safeguards for evaluation scores."""

import numpy as np
import pytest

from hypernet.partial_assignment import partial_assignment


def exhaustive_best(gains, row=0, used=frozenset()):
    if row == len(gains):
        return 0.0
    best = exhaustive_best(gains, row + 1, used)
    for col in range(gains.shape[1]):
        if col not in used:
            best = max(best, gains[row, col] + exhaustive_best(gains, row + 1, used | {col}))
    return best


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3)])
def test_assignment_agrees_with_independent_exhaustive_search(shape):
    rng = np.random.default_rng(80 + sum(shape))
    for _ in range(12):
        similarity = rng.uniform(-1, 1, shape)
        penalty = rng.uniform(0, 0.3, shape)
        weights = rng.uniform(0.01, 2, shape[0])
        weights /= weights.sum()
        expected = exhaustive_best((similarity - 0.2 - penalty) * weights[:, None])
        got = partial_assignment(similarity, weights, min_coverage=0, geometry_penalty=penalty)
        assert got["total_gain"] == pytest.approx(expected)
        assert got["score"] == pytest.approx(expected / 0.8)
        assert len(got["rows"]) == len(set(got["rows"]))
        assert len(got["cols"]) == len(set(got["cols"]))


def test_one_query_part_cannot_explain_repeated_support_parts():
    got = partial_assignment(np.array([[1.0, 0], [1.0, 0], [1.0, 0]]))
    assert len(got["cols"]) == 1
    assert got["coverage"] == pytest.approx(1 / 3)
    assert got["ungated_score"] == pytest.approx(1 / 3)
    assert got["score"] == 0
    assert not got["coverage_passed"]


def test_partial_matching_does_not_renormalize_by_selected_rows():
    got = partial_assignment([[1, 0], [0, -1]], row_weights=[3, 1], min_coverage=0.6)
    assert got["rows"] == [0]
    assert got["coverage"] == pytest.approx(0.75)
    assert got["score"] == pytest.approx(0.75)
    assert got["matched_row_weights"] == pytest.approx([0.75])
    assert got["assigned_scores"] == [1.0]


def test_geometry_can_reject_an_otherwise_strong_match():
    got = partial_assignment(np.eye(2), geometry_penalty=np.eye(2))
    assert got["rows"] == [] and got["score"] == 0


def test_perfect_rectangular_match_scores_full_support_mass():
    got = partial_assignment([[1, 0, 0], [0, 1, 0]], row_weights=[1e308, 1e308])
    assert got["score"] == pytest.approx(1)
    assert got["coverage"] == pytest.approx(1)


@pytest.mark.parametrize("matrix", [np.empty((0, 0)), np.empty((0, 3)), np.empty((3, 0)),
                                    np.full((2, 3), np.nan), np.full((3, 2), -np.inf),
                                    np.full((2, 2), 0.2)])
def test_no_evidence_returns_finite_zero(matrix):
    got = partial_assignment(matrix)
    assert got["score"] == 0 and got["coverage"] == 0
    assert got["rows"] == [] and got["cols"] == []
    assert np.isfinite(got["total_gain"])


def test_forbidden_edge_and_zero_weight_are_not_counted():
    got = partial_assignment([[1, np.nan], [1, 1]], [0, 1], geometry_penalty=[[0, 0], [np.inf, 0]])
    assert got["rows"] == [1] and got["cols"] == [1]
    assert got["score"] == 1
    assert partial_assignment(np.eye(2), [0, 0])["score"] == 0


def test_similarity_overshoot_is_bounded_and_unmatched_zero_edges_are_ignored():
    assert partial_assignment([[1.0 + 1e-8]])["score"] == 1
    assert partial_assignment([[0.2]], min_coverage=0)["rows"] == []


@pytest.mark.parametrize("kwargs", [dict(min_similarity=1), dict(min_similarity=float("nan")),
                                    dict(min_coverage=-0.1), dict(min_coverage=1.1),
                                    dict(row_weights=[1, 2]), dict(row_weights=[-1]),
                                    dict(row_weights=[float("inf")]), dict(geometry_penalty=-1)])
def test_invalid_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        partial_assignment([[1.0]], **kwargs)


def test_nonmatrix_input_is_rejected():
    with pytest.raises(ValueError):
        partial_assignment([1.0])
