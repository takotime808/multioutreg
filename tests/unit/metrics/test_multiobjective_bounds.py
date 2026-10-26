# Copyright (c) 2026 takotime808
"""Bounds, invariants and failure points of the multi-objective regret metrics."""

import numpy as np
import pytest

from multioutreg.metrics.multiobjective_regret import (
    RegretTracker,
    epsilon_regret,
    hypervolume,
    hypervolume_regret,
    pareto_mask,
    scalarized_regret,
)


@pytest.fixture
def cloud():
    return np.random.default_rng(0).uniform(size=(40, 2))


# ---------------------------------------------------------------------------
# Pareto mask
# ---------------------------------------------------------------------------


def test_pareto_mask_empty_and_single():
    assert pareto_mask(np.zeros((0, 2))).shape == (0,)
    assert pareto_mask(np.array([[1.0, 2.0]])).tolist() == [True]


def test_pareto_mask_duplicates_are_both_kept():
    Y = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    assert pareto_mask(Y).tolist() == [True, True, False]


def test_pareto_mask_1d_keeps_only_minimum_ties():
    assert pareto_mask(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [False, True, True, False]


def test_pareto_front_points_do_not_dominate_each_other(cloud):
    F = cloud[pareto_mask(cloud)]
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j:
                assert not (np.all(F[i] <= F[j]) and np.any(F[i] < F[j]))


def test_pareto_mask_maximize_equals_mask_of_negated(cloud):
    np.testing.assert_array_equal(pareto_mask(cloud, minimize=False), pareto_mask(-cloud))


def test_pareto_mask_mixed_direction_list(cloud):
    np.testing.assert_array_equal(
        pareto_mask(cloud, minimize=[True, False]),
        pareto_mask(cloud * np.array([1.0, -1.0])),
    )


@pytest.mark.parametrize("minimize", [[True], [True, True, True], [[True, True]]])
def test_minimize_with_wrong_length_raises(cloud, minimize):
    with pytest.raises(ValueError, match="minimize"):
        pareto_mask(cloud, minimize=minimize)


def test_three_dimensional_input_raises():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        pareto_mask(np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# Hypervolume bounds
# ---------------------------------------------------------------------------


def test_hypervolume_bounded_by_reference_box(cloud):
    ref = np.array([1.0, 1.0])
    hv = hypervolume(cloud, reference_point=ref)
    ideal = cloud.min(axis=0)
    assert 0.0 <= hv <= np.prod(ref - ideal) + 1e-12


def test_hypervolume_is_monotone_under_adding_points(cloud):
    ref = np.array([1.5, 1.5])
    hv_prev = 0.0
    for k in range(1, len(cloud) + 1):
        hv = hypervolume(cloud[:k], reference_point=ref)
        assert hv >= hv_prev - 1e-12
        hv_prev = hv


def test_hypervolume_grows_with_a_worse_reference_point(cloud):
    assert hypervolume(cloud, reference_point=[2.0, 2.0]) > hypervolume(cloud, reference_point=[1.0, 1.0])


def test_hypervolume_zero_when_reference_dominates_everything(cloud):
    assert hypervolume(cloud, reference_point=[-1.0, -1.0]) == 0.0


def test_hypervolume_point_on_reference_boundary_adds_nothing():
    assert hypervolume(np.array([[1.0, 0.0]]), reference_point=[1.0, 1.0]) == 0.0


def test_hypervolume_empty_with_reference_is_zero():
    assert hypervolume(np.zeros((0, 2)), reference_point=[1.0, 1.0]) == 0.0


def test_hypervolume_empty_without_reference_raises():
    # No points means no default reference point can be derived.
    with pytest.raises(ValueError):
        hypervolume(np.zeros((0, 2)))


def test_hypervolume_nan_row_is_ignored():
    Y = np.array([[np.nan, 1.0], [0.0, 0.0]])
    assert hypervolume(Y, reference_point=[2.0, 2.0]) == pytest.approx(4.0)


def test_hypervolume_reference_dimension_mismatch_raises(cloud):
    with pytest.raises(ValueError, match="dimensionality"):
        hypervolume(cloud, reference_point=[1.0, 1.0, 1.0])


def test_hypervolume_is_invariant_to_row_order(cloud):
    perm = np.random.default_rng(1).permutation(len(cloud))
    assert hypervolume(cloud, [1.2, 1.2]) == pytest.approx(hypervolume(cloud[perm], [1.2, 1.2]))


def test_hypervolume_scales_with_product_of_axis_scales(cloud):
    s = np.array([2.0, 5.0])
    ref = np.array([1.1, 1.1])
    assert hypervolume(cloud * s, ref * s) == pytest.approx(10.0 * hypervolume(cloud, ref))


# ---------------------------------------------------------------------------
# Regret bounds
# ---------------------------------------------------------------------------


def test_hypervolume_regret_is_zero_for_identical_and_non_negative_otherwise(cloud):
    assert hypervolume_regret(cloud, cloud) == 0.0
    worse = cloud + 0.3
    assert hypervolume_regret(cloud, worse) > 0
    # A better predicted front is clipped to zero regret, never negative.
    assert hypervolume_regret(worse, cloud) == 0.0


def test_hypervolume_regret_reference_dimension_mismatch_raises(cloud):
    with pytest.raises(ValueError, match="dimensionality"):
        hypervolume_regret(cloud, cloud, reference_point=[1.0])


def test_epsilon_regret_equals_uniform_shift(cloud):
    assert epsilon_regret(cloud, cloud + 0.25) == pytest.approx(0.25)
    assert epsilon_regret(cloud, cloud - 0.25) == 0.0


@pytest.mark.parametrize("shape_t, shape_p", [((0, 2), (3, 2)), ((3, 2), (0, 2))])
def test_epsilon_regret_empty_front_is_infinite(shape_t, shape_p):
    assert epsilon_regret(np.zeros(shape_t), np.zeros(shape_p)) == np.inf


def test_epsilon_regret_1d_empty_is_infinite():
    assert epsilon_regret(np.zeros(0), np.array([1.0])) == np.inf


def test_scalarized_regret_zero_for_identical(cloud):
    W = np.random.default_rng(2).dirichlet(np.ones(2), size=8)
    np.testing.assert_allclose(scalarized_regret(cloud, cloud, W, reduce="none"), 0.0)


def test_scalarized_regret_negative_weights_are_clipped():
    Yt = np.array([[0.0, 0.0]])
    Yp = np.array([[1.0, 0.0]])
    # [-1, 1] -> clip [0, 1]: objective 0 does not count.
    assert scalarized_regret(Yt, Yp, [-1.0, 1.0]) == pytest.approx(0.0)


def test_scalarized_regret_all_zero_weights_become_uniform():
    Yt, Yp = np.array([[0.0, 0.0]]), np.array([[1.0, 3.0]])
    assert scalarized_regret(Yt, Yp, [0.0, 0.0]) == pytest.approx(
        scalarized_regret(Yt, Yp, [0.5, 0.5])
    )


def test_scalarized_regret_is_invariant_to_weight_scale():
    Yt, Yp = np.array([[0.0, 0.0]]), np.array([[1.0, 3.0]])
    assert scalarized_regret(Yt, Yp, [1.0, 3.0]) == pytest.approx(scalarized_regret(Yt, Yp, [10.0, 30.0]))


def test_scalarized_regret_dict_percentiles_are_ordered(cloud):
    W = np.random.default_rng(3).dirichlet(np.ones(2), size=20)
    d = scalarized_regret(cloud, cloud * 1.2 + 0.1, W, reduce="dict")
    assert d["p10"] <= d["median"] <= d["p90"] <= d["max"]


@pytest.mark.parametrize(
    "weights, match",
    [
        (1.0, "Scalar weight"),
        ([1.0, 2.0, 3.0], "length"),
        (np.ones((2, 3)), "second dimension"),
        (np.ones((1, 1, 2)), "scalar, 1-D, or 2-D"),
    ],
)
def test_scalarized_regret_bad_weights_raise(cloud, weights, match):
    with pytest.raises(ValueError, match=match):
        scalarized_regret(cloud, cloud, weights)


def test_scalarized_regret_unknown_reduce_raises(cloud):
    with pytest.raises(ValueError, match="reduce"):
        scalarized_regret(cloud, cloud, [0.5, 0.5], reduce="sum")


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


def test_tracker_objective_count_mismatch_raises():
    with pytest.raises(ValueError, match="same number of objectives"):
        RegretTracker().step(np.zeros((3, 2)), np.zeros((3, 3)))


def test_tracker_samples_weights_on_the_simplex_once():
    t = RegretTracker(n_weight_samples=5)
    t.step(np.random.default_rng(4).uniform(size=(6, 3)), np.random.default_rng(5).uniform(size=(6, 3)))
    W = t.weights.copy()
    assert W.shape == (5, 3)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    assert np.all(W >= 0)
    t.step(np.zeros((2, 3)), np.ones((2, 3)))
    np.testing.assert_array_equal(t.weights, W)


def test_tracker_history_values_are_non_negative(cloud):
    t = RegretTracker()
    for shift in (0.5, 0.2, 0.0):
        t.step(cloud, cloud + shift)
    assert all(v >= 0 for v in t.history["hv"] + t.history["eps"])
    assert t.history["eps"][-1] == 0.0
