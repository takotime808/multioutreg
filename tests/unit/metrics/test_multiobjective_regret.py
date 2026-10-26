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


# --------------------------- exact hypervolume, any dimension ---------------------------

def _hv_inclusion_exclusion(F, ref):
    # Exact volume of the union of boxes [f, ref] by inclusion-exclusion (small fronts only).
    from itertools import combinations
    F = F[np.all(F < ref, axis=1)]
    total = 0.0
    for k in range(1, len(F) + 1):
        for subset in combinations(range(len(F)), k):
            corner = F[list(subset)].max(axis=0)
            total += (-1) ** (k + 1) * np.prod(ref - corner)
    return total


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5])
def test_hypervolume_single_point_is_box_volume(m):
    from multioutreg.metrics import hypervolume
    point = np.linspace(0.1, 0.5, m)[None, :]
    ref = np.ones(m)
    assert hypervolume(point, reference_point=ref) == pytest.approx(np.prod(ref - point[0]))


@pytest.mark.parametrize("m,n,seed", [(3, 6, 0), (3, 10, 1), (4, 7, 2), (4, 9, 3), (5, 6, 4)])
def test_hypervolume_matches_inclusion_exclusion(m, n, seed):
    from multioutreg.metrics import hypervolume
    rng = np.random.default_rng(seed)
    F = rng.random((n, m))
    ref = np.full(m, 1.1)
    assert hypervolume(F, reference_point=ref) == pytest.approx(_hv_inclusion_exclusion(F, ref), rel=1e-12)


def test_hypervolume_2d_staircase_by_hand():
    from multioutreg.metrics import hypervolume
    Y = np.array([[1.0, 4.0], [2.0, 2.5], [3.0, 2.0], [4.0, 1.0], [3.5, 3.5], [2.5, 4.5]])
    # rectangles: 4*1 + 3*1.5 + 2*0.5 + 1*1
    assert hypervolume(Y, reference_point=[5.0, 5.0]) == pytest.approx(10.5)


def test_hypervolume_ignores_dominated_and_out_of_reference_points():
    from multioutreg.metrics import hypervolume
    F = np.array([[0.2, 0.2, 0.2], [0.5, 0.1, 0.6]])
    ref = np.ones(3)
    base = hypervolume(F, reference_point=ref)
    extra = np.vstack([F, [[0.3, 0.3, 0.3]], [[0.05, 0.05, 1.5]], [[2.0, 0.0, 0.0]]])
    assert hypervolume(extra, reference_point=ref) == pytest.approx(base)
    assert hypervolume(np.array([[2.0, 2.0, 2.0]]), reference_point=ref) == 0.0


def test_hypervolume_maximization_and_default_reference():
    from multioutreg.metrics import hypervolume
    rng = np.random.default_rng(5)
    F = rng.random((8, 3))
    ref = np.full(3, 1.2)
    flip = np.array([1.0, -1.0, 1.0])
    # Maximizing a negated objective, with the reference given in the minimization convention.
    assert hypervolume(F * flip, reference_point=ref, minimize=[True, False, True]) == pytest.approx(
        hypervolume(F, reference_point=ref))
    default_ref = F.max(0) + 0.1 * np.ptp(F, 0)
    assert hypervolume(F) == pytest.approx(hypervolume(F, reference_point=default_ref))
    with pytest.raises(ValueError):
        hypervolume(F, reference_point=[1.0, 1.0])


def test_hypervolume_regret_3d_is_exact():
    rng = np.random.default_rng(6)
    Y_true = rng.random((10, 3))
    ref = np.full(3, 1.5)
    assert hypervolume_regret(Y_true, Y_true, reference_point=ref) == 0.0
    Y_pred = Y_true + 0.1
    expected = _hv_inclusion_exclusion(Y_true[pareto_mask(Y_true)], ref) - _hv_inclusion_exclusion(
        Y_pred[pareto_mask(Y_pred)], ref)
    assert hypervolume_regret(Y_true, Y_pred, reference_point=ref) == pytest.approx(expected, rel=1e-12)


def test_hypervolume_agrees_with_pymoo_when_installed():
    pytest.importorskip("pymoo")
    from multioutreg.metrics import hypervolume
    from multioutreg.metrics.multiobjective_regret import _hv_exact_min
    rng = np.random.default_rng(7)
    F = rng.random((40, 4))
    ref = np.full(4, 1.1)
    assert hypervolume(F, reference_point=ref) == pytest.approx(_hv_exact_min(F, ref), rel=1e-9)


# --------------------------- input validation & helpers ---------------------------

from multioutreg.metrics import hypervolume  # noqa: E402
from multioutreg.metrics import multiobjective_regret as mor  # noqa: E402


def test_package_exports_match_module():
    import multioutreg.metrics as pkg
    assert set(pkg.__all__) == {
        "pareto_mask", "hypervolume", "hypervolume_regret",
        "scalarized_regret", "epsilon_regret", "RegretTracker",
    }
    for name in pkg.__all__:
        assert getattr(pkg, name) is getattr(mor, name)


def test_ensure_2d_reshapes_1d_and_rejects_3d():
    out = mor._ensure_2d([1, 2, 3])
    assert out.shape == (3, 1) and out.dtype == float
    Y = np.ones((4, 2))
    assert mor._ensure_2d(Y).shape == (4, 2)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        mor._ensure_2d(np.ones((2, 2, 2)))


def test_as_minimization_bool_and_per_objective():
    Y = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert np.array_equal(mor._as_minimization(Y, True), Y)
    assert np.array_equal(mor._as_minimization(Y, False), -Y)
    assert np.array_equal(mor._as_minimization(Y, [True, False]), Y * [1.0, -1.0])
    with pytest.raises(ValueError, match="minimize"):
        mor._as_minimization(Y, [True, False, True])
    with pytest.raises(ValueError, match="minimize"):
        mor._as_minimization(Y, [[True, False]])


def test_pareto_mask_empty_single_and_duplicates():
    assert pareto_mask(np.empty((0, 2))).shape == (0,)
    assert pareto_mask(np.array([[3.0, 4.0]])).tolist() == [True]
    # Identical points do not dominate each other.
    assert pareto_mask(np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])).tolist() == [True, True, False]


def test_pareto_mask_1d_and_maximization():
    assert pareto_mask(np.array([3.0, 1.0, 2.0, 1.0])).tolist() == [False, True, False, True]
    Y = np.array([[1.0, 5.0], [2.0, 2.0], [4.0, 4.0]])
    # Maximize both: [4,4] dominates [2,2]; [1,5] survives on objective 2.
    assert pareto_mask(Y, minimize=False).tolist() == [True, False, True]
    # Minimize first, maximize second: [1,5] dominates everything.
    assert pareto_mask(Y, minimize=[True, False]).tolist() == [True, False, False]


def test_default_reference_point_margin_and_constant_dims():
    A = np.array([[0.0, 5.0], [2.0, 5.0]])
    B = np.array([[1.0, 5.0]])
    ref = mor._default_reference_point([A, B])
    # dim 0: worst 2 + 10% of span 2; dim 1 is constant -> worst + 1.0
    assert np.allclose(ref, [2.2, 6.0])
    assert np.allclose(mor._default_reference_point([np.array([1.0, 3.0])]), [3.2])


def test_hypervolume_1d_helper():
    ref = np.array([3.0])
    assert mor._hypervolume_1d_min(np.empty((0, 1)), ref) == 0.0
    assert mor._hypervolume_1d_min(np.array([[1.0], [0.5]]), ref) == pytest.approx(2.5)
    assert mor._hypervolume_1d_min(np.array([[4.0]]), ref) == 0.0


def test_hv_2d_helper_empty_and_outside_points():
    ref = np.array([2.0, 2.0])
    assert mor._hv_2d_min(np.empty((0, 2)), ref) == 0.0
    # A point beyond the reference in x contributes zero width.
    assert mor._hv_2d_min(np.array([[3.0, 0.0]]), ref) == 0.0
    # Unsorted input is sorted internally; dominated point adds nothing.
    F = np.array([[1.0, 0.0], [0.0, 1.0], [1.5, 1.5]])
    assert mor._hv_2d_min(F, ref) == pytest.approx(3.0)


def test_hv_exact_tied_last_objective_and_1d():
    # Two front points share the last objective -> a zero-depth slab is skipped.
    F = np.array([[0.2, 0.5, 0.3], [0.5, 0.2, 0.3], [0.1, 0.9, 0.6]])
    ref = np.ones(3)
    assert mor._hv_exact_min(F, ref) == pytest.approx(_hv_inclusion_exclusion(F, ref), rel=1e-12)
    assert mor._hv_exact_min(np.array([[0.25], [0.5]]), np.array([1.0])) == pytest.approx(0.75)
    assert mor._hv_exact_min(np.array([[1.0, 0.0, 0.0]]), ref) == 0.0  # on the boundary


def test_hypervolume_1d_input_and_default_reference():
    assert hypervolume(np.array([1.0, 2.0, 3.0]), reference_point=[4.0]) == pytest.approx(3.0)
    # default ref = 3 + 0.1 * 2 = 3.2
    assert hypervolume(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.2)


def test_hypervolume_empty_and_nothing_inside_reference():
    assert hypervolume(np.empty((0, 2)), reference_point=[1.0, 1.0]) == 0.0
    assert hypervolume(np.empty((0, 3)), reference_point=[1.0, 1.0, 1.0]) == 0.0
    assert hypervolume(np.array([[2.0, 0.0]]), reference_point=[1.0, 1.0]) == 0.0


# --------------------------- pymoo branch (stubbed) ---------------------------

@pytest.fixture
def fake_pymoo(monkeypatch):
    """Install a stand-in ``pymoo.indicators.hv`` that records calls and uses the exact HV."""
    import sys
    import types

    calls = []

    class HV:
        def __init__(self, ref_point):
            self.ref_point = np.asarray(ref_point, float)

        def do(self, F):
            calls.append((self.ref_point.copy(), np.asarray(F).copy()))
            return mor._hv_exact_min(np.asarray(F, float), self.ref_point)

    pymoo = types.ModuleType("pymoo")
    indicators = types.ModuleType("pymoo.indicators")
    hv_mod = types.ModuleType("pymoo.indicators.hv")
    hv_mod.HV = HV
    pymoo.indicators = indicators
    indicators.hv = hv_mod
    monkeypatch.setitem(sys.modules, "pymoo", pymoo)
    monkeypatch.setitem(sys.modules, "pymoo.indicators", indicators)
    monkeypatch.setitem(sys.modules, "pymoo.indicators.hv", hv_mod)
    return calls


def test_hypervolume_uses_pymoo_for_3plus_objectives(fake_pymoo):
    F = np.array([[0.2, 0.2, 0.2], [0.5, 0.1, 0.6], [2.0, 0.0, 0.0]])
    ref = np.ones(3)
    hv = hypervolume(F, reference_point=ref)
    assert len(fake_pymoo) == 1
    ref_used, F_used = fake_pymoo[0]
    assert np.array_equal(ref_used, ref)
    # Only points strictly inside the reference box are handed to pymoo.
    assert F_used.shape == (2, 3)
    assert hv == pytest.approx(_hv_inclusion_exclusion(F, ref), rel=1e-12)


def test_hypervolume_pymoo_branch_returns_zero_without_inside_points(fake_pymoo):
    assert hypervolume(np.array([[2.0, 2.0, 2.0]]), reference_point=np.ones(3)) == 0.0
    assert fake_pymoo == []


def test_hypervolume_2d_does_not_use_pymoo(fake_pymoo):
    hypervolume(np.array([[0.0, 1.0], [1.0, 0.0]]), reference_point=[2.0, 2.0])
    assert fake_pymoo == []


def test_hypervolume_falls_back_when_pymoo_missing(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "pymoo", None)  # makes the import raise ImportError
    monkeypatch.setitem(sys.modules, "pymoo.indicators", None)
    monkeypatch.setitem(sys.modules, "pymoo.indicators.hv", None)
    F = np.random.default_rng(8).random((6, 3))
    ref = np.full(3, 1.1)
    assert hypervolume(F, reference_point=ref) == pytest.approx(_hv_inclusion_exclusion(F, ref), rel=1e-12)


# --------------------------- hypervolume_regret ---------------------------

def test_hypervolume_regret_equals_hv_difference():
    rng = np.random.default_rng(9)
    Yt = rng.random((12, 2))
    Yp = Yt + rng.normal(0.0, 0.05, Yt.shape)
    ref = np.full(2, 1.5)
    expected = hypervolume(Yt, reference_point=ref) - hypervolume(Yp, reference_point=ref)
    assert hypervolume_regret(Yt, Yp, reference_point=ref) == pytest.approx(max(0.0, expected))


def test_hypervolume_regret_clipped_at_zero_when_pred_is_better():
    Yt = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert hypervolume_regret(Yt, Yt - 0.5, reference_point=[3.0, 3.0]) == 0.0


def test_hypervolume_regret_default_reference_and_validation():
    Yt = np.array([[0.0, 1.0], [1.0, 0.0]])
    Yp = np.array([[0.5, 1.5], [1.5, 0.5]])
    ref = mor._default_reference_point([Yt, Yp])
    assert hypervolume_regret(Yt, Yp) == pytest.approx(hypervolume_regret(Yt, Yp, reference_point=ref))
    with pytest.raises(ValueError, match="reference_point"):
        hypervolume_regret(Yt, Yp, reference_point=[1.0, 2.0, 3.0])


def test_hypervolume_regret_maximization_and_empty_pred():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = Yt - 0.3  # worse when maximizing
    r = hypervolume_regret(Yt, Yp, reference_point=[1.0, 1.0], minimize=False)
    assert r == pytest.approx(hypervolume_regret(-Yt, -Yp, reference_point=[1.0, 1.0]))
    assert r > 0
    # No predicted points: regret is the whole true hypervolume.
    assert hypervolume_regret(Yt, np.empty((0, 2)), reference_point=[3.0, 3.0]) == pytest.approx(
        hypervolume(Yt, reference_point=[3.0, 3.0]))


# --------------------------- scalarized_regret ---------------------------

def test_scalarized_regret_scalar_weight_requires_single_objective():
    Y = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Scalar weight"):
        scalarized_regret(Y, Y, weights=1.0)


def test_scalarized_regret_2d_weights_wrong_width_and_3d_weights():
    Y = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="second dimension"):
        scalarized_regret(Y, Y, weights=np.ones((2, 3)))
    with pytest.raises(ValueError, match="scalar, 1-D, or 2-D"):
        scalarized_regret(Y, Y, weights=np.ones((1, 1, 2)))


def test_scalarized_regret_weights_are_clipped_and_normalized():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = Yt + np.array([0.2, 0.4])
    # Unnormalized and negative entries map to the same simplex vector.
    a = scalarized_regret(Yt, Yp, weights=[[2.0, 2.0]], reduce="none")
    b = scalarized_regret(Yt, Yp, weights=[[0.5, 0.5]], reduce="none")
    assert np.allclose(a, b)
    c = scalarized_regret(Yt, Yp, weights=[[-3.0, 1.0]], reduce="none")
    d = scalarized_regret(Yt, Yp, weights=[[0.0, 1.0]], reduce="none")
    assert np.allclose(c, d) and c[0] == pytest.approx(-0.4)


def test_scalarized_regret_zero_weight_rows_become_uniform():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = Yt + np.array([0.2, 0.4])
    uniform = scalarized_regret(Yt, Yp, weights=[0.5, 0.5], reduce="none")
    assert np.allclose(scalarized_regret(Yt, Yp, weights=[0.0, 0.0], reduce="none"), uniform)
    mixed = scalarized_regret(Yt, Yp, weights=[[0.0, 0.0], [1.0, 0.0]], reduce="none")
    assert mixed[0] == pytest.approx(uniform[0])
    assert mixed[1] == pytest.approx(-0.2)


def test_scalarized_regret_reduce_options():
    Yt = np.zeros((3, 2))
    Yp = np.ones((3, 2)) * np.array([1.0, 3.0])
    W = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.25, 0.75]])
    r = scalarized_regret(Yt, Yp, weights=W, reduce="none")
    assert np.allclose(r, [-1.0, -2.0, -3.0, -2.5])
    assert scalarized_regret(Yt, Yp, weights=W, reduce="mean") == pytest.approx(np.mean(r))
    assert scalarized_regret(Yt, Yp, weights=W, reduce="median") == pytest.approx(np.median(r))
    assert scalarized_regret(Yt, Yp, weights=W, reduce="max") == pytest.approx(-1.0)
    d = scalarized_regret(Yt, Yp, weights=W, reduce="dict")
    assert set(d) == {"mean", "median", "max", "p10", "p90"}
    assert d["p10"] == pytest.approx(np.percentile(r, 10))
    assert d["p90"] == pytest.approx(np.percentile(r, 90))
    assert all(isinstance(v, float) for v in d.values())
    with pytest.raises(ValueError, match="reduce"):
        scalarized_regret(Yt, Yp, weights=W, reduce="sum")


def test_scalarized_regret_maximization():
    Yt = np.array([[3.0, 1.0], [1.0, 3.0]])
    Yp = np.array([[2.0, 1.0], [1.0, 2.0]])
    r = scalarized_regret(Yt, Yp, weights=[0.5, 0.5], minimize=False)
    # min over -w.Y: true -2, pred -1.5 -> -0.5
    assert r == pytest.approx(-0.5)


# --------------------------- epsilon_regret ---------------------------

def test_epsilon_regret_empty_fronts_are_infinite():
    Y2 = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert epsilon_regret(Y2, np.empty((0, 2))) == np.inf
    assert epsilon_regret(np.empty((0, 2)), Y2) == np.inf
    assert epsilon_regret(np.array([[1.0]]), np.empty((0, 1))) == np.inf


def test_epsilon_regret_clipped_when_pred_dominates():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    assert epsilon_regret(Yt, Yt - 0.3) == 0.0
    assert epsilon_regret(np.array([1.0, 2.0]), np.array([0.5])) == 0.0


def test_epsilon_regret_partial_coverage_and_maximization():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = np.array([[0.0, 2.0], [2.0, 0.0]])  # misses the knee point
    assert epsilon_regret(Yt, Yp) == pytest.approx(1.0)
    assert epsilon_regret(-Yt, -Yp, minimize=False) == pytest.approx(1.0)
    assert epsilon_regret(Yt, Yp, minimize=[True, True]) == pytest.approx(1.0)


# --------------------------- RegretTracker ---------------------------

def test_tracker_defaults():
    t = RegretTracker()
    assert t.minimize is True and t.reference_point is None and t.weights is None
    assert t.n_weight_samples == 16
    assert t.history == {"hv": [], "scalar": [], "eps": []}
    # Separate instances do not share history.
    assert RegretTracker().history is not t.history


def test_tracker_samples_reproducible_simplex_weights_once():
    t = RegretTracker(n_weight_samples=5)
    Y = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0]])
    t.step(Y, Y + 0.1)
    W = t.weights.copy()
    assert W.shape == (5, 3)
    assert np.all(W >= 0) and np.allclose(W.sum(axis=1), 1.0)
    t.step(Y, Y + 0.2)
    assert np.array_equal(t.weights, W)  # not resampled
    t2 = RegretTracker(n_weight_samples=5)
    t2.step(Y, Y)
    assert np.array_equal(t2.weights, W)  # fixed seed


def test_tracker_keeps_user_weights_and_reference_point():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = Yt + 0.25
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    ref = np.array([4.0, 4.0])
    t = RegretTracker(reference_point=ref, weights=W)
    out = t.step(Yt, Yp)
    assert t.weights is W
    assert out["hv"] == pytest.approx(hypervolume_regret(Yt, Yp, reference_point=ref))
    assert out["scalar"] == pytest.approx(scalarized_regret(Yt, Yp, weights=W))
    assert out["eps"] == pytest.approx(epsilon_regret(Yt, Yp))
    assert out["eps"] == pytest.approx(0.25)


def test_tracker_default_reference_matches_functional_api():
    Yt = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    Yp = Yt + np.array([0.1, 0.3])
    out = RegretTracker().step(Yt, Yp)
    assert out["hv"] == pytest.approx(hypervolume_regret(Yt, Yp))


def test_tracker_mixed_minimize_and_1d_inputs():
    Yt = np.array([[1.0, 5.0], [2.0, 6.0]])
    Yp = np.array([[1.5, 4.0], [2.5, 5.0]])
    t = RegretTracker(minimize=[True, False])
    out = t.step(Yt, Yp)
    ref = mor._default_reference_point([Yt * [1, -1], Yp * [1, -1]])
    assert out["hv"] == pytest.approx(
        hypervolume_regret(Yt, Yp, reference_point=ref, minimize=[True, False]))
    assert out["eps"] == pytest.approx(epsilon_regret(Yt, Yp, minimize=[True, False]))

    t1 = RegretTracker()
    res = t1.step([0.2, 0.5], [0.4])
    assert t1.weights.shape == (16, 1)
    assert res["eps"] == pytest.approx(0.2)
    assert res["scalar"] == pytest.approx(-0.2)


def test_tracker_rejects_mismatched_objectives():
    t = RegretTracker()
    with pytest.raises(ValueError, match="same number of objectives"):
        t.step(np.ones((3, 2)), np.ones((3, 3)))
    assert t.history == {"hv": [], "scalar": [], "eps": []}


def test_tracker_history_matches_step_outputs():
    rng = np.random.default_rng(10)
    t = RegretTracker(n_weight_samples=4)
    outs = [t.step(Y, Y + rng.random(Y.shape) * 0.1) for Y in rng.random((3, 8, 2))]
    for key in ("hv", "scalar", "eps"):
        assert t.history[key] == [o[key] for o in outs]
        assert all(isinstance(v, float) for v in t.history[key])
