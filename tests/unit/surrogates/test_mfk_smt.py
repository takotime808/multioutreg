# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.base import clone

from multioutreg.active.benchmarks import forrester, forrester_lo, get_benchmark
from multioutreg.surrogates import mfk_smt

pytestmark = pytest.mark.skipif(not mfk_smt._MFK_AVAILABLE, reason="smt not installed")


def _forrester_data(n_lo=20, n_hi=5, seed=0):
    rng = np.random.default_rng(seed)
    X_lo, X_hi = rng.random((n_lo, 1)), rng.random((n_hi, 1))  # not nested
    return {"lo": (X_lo, forrester_lo(X_lo)), "hi": (X_hi, forrester(X_hi))}


def test_recovers_forrester_rho_from_five_high_fidelity_points():
    # forrester = 2 * forrester_lo - 20 x + 20, so the true scale factor is 2;
    # AdditiveCorrectionVFM's least-squares rho gets 0.44 on this pair.
    model = mfk_smt.MFKSurrogate(n_start=3).fit(_forrester_data())
    assert model.rho_.shape == (1, 1)
    assert model.rho_[0, 0] == pytest.approx(2.0, abs=0.05)
    X = np.linspace(0, 1, 200).reshape(-1, 1)
    rmse = np.sqrt(np.mean((model.predict(X) - forrester(X)) ** 2))
    assert rmse < 0.1  # the function spans about [-6, 16]


def test_lower_level_prediction_and_std():
    model = mfk_smt.MFKSurrogate(n_start=3).fit(_forrester_data())
    X = np.linspace(0, 1, 50).reshape(-1, 1)
    lo, lo_std = model.predict(X, return_std=True, level="lo")
    hi, hi_std = model.predict(X, return_std=True)
    np.testing.assert_allclose(lo, forrester_lo(X), atol=0.05)
    assert lo.shape == lo_std.shape == hi.shape == hi_std.shape == (50, 1)
    assert np.all(lo_std >= 0) and np.all(hi_std >= 0)
    with pytest.raises(ValueError, match="unknown fidelity level"):
        model.predict(X, level="mid")


def test_three_levels_and_multiple_outputs():
    rng = np.random.default_rng(1)

    def f(X, a):  # two outputs per level, scaled copies across levels
        return np.column_stack([a * np.sin(4 * X[:, 0]), a * np.cos(3 * X[:, 0]) + X[:, 0]])

    data = {}
    for lvl, n, a in [("lo", 30, 0.5), ("mid", 15, 0.8), ("hi", 6, 1.0)]:
        X = rng.random((n, 1))
        data[lvl] = (X, f(X, a))
    model = mfk_smt.MFKSurrogate(["lo", "mid", "hi"], n_start=3).fit(data)
    assert model.rho_.shape == (2, 2)
    X = np.linspace(0, 1, 40).reshape(-1, 1)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == std.shape == (40, 2)
    assert np.sqrt(np.mean((mean - f(X, 1.0)) ** 2)) < 0.1


def test_pls_variant_on_high_dimensional_inputs():
    bench = get_benchmark("borehole")
    X_lo, X_hi = bench.sample(40, seed=0), bench.sample(12, seed=1)
    data = {"lo": (X_lo, bench(X_lo, "lo")), "hi": (X_hi, bench(X_hi, "hi"))}
    model = mfk_smt.MFKSurrogate(n_comp=2, n_start=2).fit(data)
    X = bench.sample(30, seed=2)
    pred = model.predict(X)[:, 0]
    truth = bench(X, "hi")[:, 0]
    assert np.mean((pred - truth) ** 2) < np.var(truth)


def test_linear_rho_has_no_scalar_rho():
    model = mfk_smt.MFKSurrogate(rho_regr="linear", n_start=2).fit(_forrester_data())
    assert model.rho_ is None
    assert model.predict(np.array([[0.5]])).shape == (1, 1)


def test_validation_errors_and_clone():
    with pytest.raises(ValueError, match="at least two"):
        mfk_smt.MFKSurrogate(["hi"])
    data = _forrester_data()
    with pytest.raises(ValueError, match="no data"):
        mfk_smt.MFKSurrogate(["lo", "mid", "hi"]).fit(data)
    bad = {"lo": data["lo"], "hi": (data["hi"][0], np.column_stack([data["hi"][1]] * 2))}
    with pytest.raises(ValueError, match="same number of outputs"):
        mfk_smt.MFKSurrogate().fit(bad)
    with pytest.raises(AttributeError):
        mfk_smt.MFKSurrogate().predict(np.zeros((1, 1)))
    model = mfk_smt.MFKSurrogate(["a", "b"], corr="matern52")
    assert clone(model).get_params() == model.get_params()
