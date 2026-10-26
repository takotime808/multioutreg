# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of the sampling-method detector."""

import numpy as np
import pytest

from multioutreg.sampling.detect import (
    _mc_p_value,
    _simulate,
    default_n_sim,
    design_statistics,
    detect_sampling_method,
    grid_structure,
    to_unit_cube,
)


# ---------------------------------------------------------------------------
# Unit-cube mapping
# ---------------------------------------------------------------------------


def test_to_unit_cube_without_bounds_stays_strictly_inside():
    X = np.random.default_rng(0).normal(scale=100, size=(25, 3))
    U = to_unit_cube(X)
    n = len(X)
    np.testing.assert_allclose(U.min(axis=0), 0.5 / n)
    np.testing.assert_allclose(U.max(axis=0), 1 - 0.5 / n)


def test_to_unit_cube_constant_column_maps_to_lowest_stratum_centre():
    U = to_unit_cube(np.ones((4, 1)))
    np.testing.assert_allclose(U, 0.125)


def test_to_unit_cube_with_bounds_clips_out_of_range_points():
    U = to_unit_cube(np.array([[-1.0, 0.5], [2.0, 0.25]]), bounds=(0.0, 1.0))
    np.testing.assert_array_equal(U, [[0.0, 0.5], [1.0, 0.25]])


def test_to_unit_cube_per_column_bounds_broadcast():
    X = np.array([[5.0, 50.0], [10.0, 100.0]])
    U = to_unit_cube(X, bounds=([0.0, 0.0], [10.0, 100.0]))
    np.testing.assert_allclose(U, [[0.5, 0.5], [1.0, 1.0]])


def test_to_unit_cube_zero_width_bounds_raises():
    with pytest.raises(ValueError):
        to_unit_cube(np.array([[1.0]]), bounds=(1.0, 1.0))


# ---------------------------------------------------------------------------
# Statistics and p-values
# ---------------------------------------------------------------------------


def test_design_statistics_are_bounded():
    U = np.random.default_rng(1).random((40, 3))
    s = design_statistics(U)
    assert 0 < s["strata"] <= 1
    assert s["nn"] > 0
    assert np.isfinite(s["log_cd"])


def test_design_statistics_perfect_lhs_fills_every_stratum():
    n = 20
    perm = np.random.default_rng(2).permutation(n)
    U = np.column_stack([(np.arange(n) + 0.5) / n, (perm + 0.5) / n])
    assert design_statistics(U)["strata"] == 1.0


def test_design_statistics_duplicate_points_have_zero_nn_distance():
    U = np.repeat(np.random.default_rng(3).random((5, 2)), 2, axis=0)
    assert design_statistics(U)["nn"] == 0.0


def test_mc_p_value_bounds():
    sim = np.random.default_rng(4).normal(size=(99, 2))
    assert _mc_p_value(np.array([1e6, 1e6]), sim) == pytest.approx(1 / 100)
    assert _mc_p_value(sim.mean(axis=0), sim) == 1.0


def test_mc_p_value_constant_statistic_does_not_crash():
    sim = np.column_stack([np.ones(49), np.random.default_rng(5).normal(size=49)])
    p = _mc_p_value(np.array([1.0, 0.0]), sim)
    assert 0 < p <= 1


@pytest.mark.parametrize("n, expected", [(1, 199), (500, 199), (501, 99), (2000, 99), (2001, 49)])
def test_default_n_sim_boundaries(n, expected):
    assert default_n_sim(n) == expected


def test_simulate_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown sampling method"):
        _simulate("Halton-ish", 5, 2, np.random.default_rng(0))


@pytest.mark.parametrize("method", ["Random", "LHS", "Sobol"])
def test_simulated_designs_lie_in_unit_cube(method):
    S = _simulate(method, 16, 3, np.random.default_rng(6))
    assert S.shape == (16, 3)
    assert np.all((S >= 0) & (S < 1))


# ---------------------------------------------------------------------------
# Grid structure
# ---------------------------------------------------------------------------


def test_grid_structure_single_level_column_is_not_grid():
    X = np.column_stack([np.ones(8), np.tile([0.0, 1.0], 4)])
    grid_like, full, levels = grid_structure(X)
    assert not grid_like and not full
    assert levels.tolist() == [1, 2]


def test_grid_structure_full_factorial_with_duplicates_is_not_full():
    X = np.array([[0, 0], [0, 1], [1, 0], [0, 0]], dtype=float)
    grid_like, full, _ = grid_structure(X)
    assert grid_like and not full


# ---------------------------------------------------------------------------
# detect_sampling_method failure points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(3, 2), (0, 2), (10, 0)])
def test_detect_rejects_too_small_input(shape):
    with pytest.raises(ValueError, match="at least 4 points"):
        detect_sampling_method(np.zeros(shape))


def test_detect_rejects_none():
    with pytest.raises(ValueError, match="required"):
        detect_sampling_method(None)


def test_detect_accepts_1d_input_as_single_column():
    x = np.random.default_rng(7).random(12)
    res = detect_sampling_method(x, n_sim=19)
    assert res.method in {"Random", "LHS", "Sobol", "Uncertain"}


def test_detect_between_4_and_min_mc_points_is_uncertain():
    X = np.random.default_rng(8).random((6, 2))
    assert detect_sampling_method(X).method == "Uncertain"


def test_detect_all_non_finite_rows_raises():
    with pytest.raises(ValueError, match="without missing or infinite"):
        detect_sampling_method(np.full((6, 2), np.nan))


def test_detect_four_finite_rows_after_dropping_is_accepted():
    X = np.random.default_rng(12).random((6, 2))
    X[:2, 1] = np.inf
    assert detect_sampling_method(X).method == "Uncertain"


def test_detect_p_values_are_probabilities_and_n_sim_is_respected():
    X = np.random.default_rng(9).random((30, 2))
    res = detect_sampling_method(X, n_sim=19)
    assert res.n_sim == 19
    assert set(res.p_values) == {"Random", "LHS", "Sobol"}
    assert all(1 / 20 <= p <= 1 for p in res.p_values.values())


def test_detect_restricted_candidates():
    X = np.random.default_rng(10).random((30, 2))
    res = detect_sampling_method(X, n_sim=19, candidates=("Random",))
    assert set(res.p_values) == {"Random"}


def test_detect_alpha_above_one_rejects_every_candidate():
    X = np.random.default_rng(11).random((30, 2))
    assert detect_sampling_method(X, n_sim=19, alpha=1.01).method == "Uncertain"


def test_detect_too_few_finite_rows_raises_instead_of_matching_sobol():
    # Regression: the 4-point minimum was checked before dropping NaN rows, so
    # 2 surviving points were reported as an exact unscrambled-Sobol match.
    X = np.random.default_rng(0).random((5, 2))
    X[:3, 0] = np.nan
    with pytest.raises(ValueError, match="at least 4 rows"):
        detect_sampling_method(X)
