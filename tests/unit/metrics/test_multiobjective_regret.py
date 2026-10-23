# Copyright (c) 2025 takotime808
import numpy as np
import pytest

from multioutreg.metrics import (
    pareto_mask,
    hypervolume_regret,
    scalarized_regret,
    epsilon_regret,
    RegretTracker,
)


def test_pareto_mask_known_2d_minimization():
    # Non-dominated: [1,5], [2,2], [3,1.5]; dominated: [4,4]
    Y = np.array([
        [1.0, 5.0],
        [2.0, 2.0],
        [3.0, 1.5],
        [4.0, 4.0],
    ])
    mask = pareto_mask(Y, True)
    assert mask.tolist() == [True, True, True, False]


def test_hypervolume_regret_1d_exact_ref():
    # 1D exact HV: HV = max(0, ref - min(front))
    Y_true = np.array([[0.0], [1.0], [2.0]])
    Y_pred = np.array([[0.1], [1.2]])
    ref = np.array([3.0])

    r = hypervolume_regret(Y_true, Y_pred, reference_point=ref, minimize=True)
    # HV_true = 3.0 - 0.0 = 3.0; HV_pred = 3.0 - 0.1 = 2.9; regret = 0.1
    assert pytest.approx(r, rel=0, abs=1e-12) == 0.1


def test_hypervolume_regret_2d_identical_zero():
    Y = np.array([[0., 2.], [1., 1.], [2., 0.]])
    r = hypervolume_regret(Y, Y, reference_point=np.array([3., 3.]), minimize=True)
    assert pytest.approx(r, abs=1e-12) == 0.0


def test_scalarized_regret_shapes_and_values_m1():
    # m = 1, scalar weights should be accepted; r = min(Yt) - min(Yp)
    Yt = np.array([[0.5], [0.7], [1.2]])
    Yp = np.array([[0.6], [0.8]])
    # Expected: 0.5 - 0.6 = -0.1
    r_scalar = scalarized_regret(Yt, Yp, weights=1.0, reduce="mean")
    assert isinstance(r_scalar, float)
    assert pytest.approx(r_scalar, abs=1e-12) == -0.1

    # Also accept (k,) weights when m==1
    r_vec = scalarized_regret(Yt, Yp, weights=np.array([1.0, 2.0, 3.0]), reduce="none")
    assert r_vec.shape == (3,)
    # All entries equal to same value because m==1 (they get normalized per-row)
    assert np.allclose(r_vec, r_vec[0])


def test_scalarized_regret_m2_valid_and_invalid_weights():
    Yt = np.array([[0., 2.], [1., 1.], [2., 0.]])
    Yp = np.array([[0.5, 1.7], [1.2, 0.9], [2.2, 0.1]])

    # Valid shapes
    r_mean = scalarized_regret(Yt, Yp, weights=np.array([0.5, 0.5]), reduce="mean")
    assert isinstance(r_mean, float)

    r_none = scalarized_regret(
        Yt, Yp, weights=np.array([[0.5, 0.5], [0.2, 0.8]]), reduce="none"
    )
    assert r_none.shape == (2,)

    # Invalid: 1-D weights length != m should raise
    with pytest.raises(ValueError):
        scalarized_regret(Yt, Yp, weights=np.array([1.0, 2.0, 3.0]), reduce="mean")


def test_epsilon_regret_identical_zero_and_shifted_positive():
    # Identical fronts -> epsilon 0
    Ft = np.array([[0., 2.], [1., 1.], [2., 0.]])
    assert pytest.approx(epsilon_regret(Ft, Ft, True), abs=1e-12) == 0.0

    # If predicted front is uniformly worse by +0.2 in all dims, eps = 0.2
    Fp = Ft + 0.2
    eps = epsilon_regret(Ft, Fp, True)
    assert pytest.approx(eps, abs=1e-12) == 0.2


def test_epsilon_regret_1d_exact():
    Ft = np.array([[0.2], [0.5], [1.0]])
    Fp = np.array([[0.6], [0.8]])
    # 1D: eps = min(Fp) - min(Ft), clipped to >=0 = 0.6 - 0.2 = 0.4
    eps = epsilon_regret(Ft, Fp, True)
    assert pytest.approx(eps, abs=1e-12) == 0.4


def test_tracker_runs_and_accumulates_m2():
    tracker = RegretTracker(minimize=True, n_weight_samples=4)
    Yt1 = np.array([[0.0, 2.0], [1.0, 1.0]])
    Yp1 = np.array([[0.4, 1.8], [1.2, 0.9]])
    out1 = tracker.step(Yt1, Yp1)
    assert set(out1.keys()) == {"hv", "scalar", "eps"}
    assert len(tracker.history["hv"]) == 1

    Yt2 = np.array([[0.2, 1.9], [1.0, 0.95]])
    Yp2 = np.array([[0.5, 1.6], [1.1, 0.8]])
    tracker.step(Yt2, Yp2)
    assert len(tracker.history["hv"]) == 2


def test_tracker_runs_m1():
    tracker = RegretTracker(minimize=True, n_weight_samples=3)
    Yt = np.array([[0.5], [0.9], [1.1]])
    Yp = np.array([[0.6], [1.0]])
    res = tracker.step(Yt, Yp)
    assert {"hv", "scalar", "eps"} <= set(res.keys())
    assert len(tracker.history["scalar"]) == 1
