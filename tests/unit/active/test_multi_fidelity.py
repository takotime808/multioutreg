# Copyright (c) 2026 takotime808

import warnings

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from multioutreg.active import (
    MultiFidelityActiveLearner,
    effective_costs,
    fidelity_info,
    get_benchmark,
    level_trust,
)
from multioutreg.surrogates import (
    AdditiveCorrectionVFM,
    GaussianProcessSurrogate,
    StackedVFMSurrogate,
)


def gp():
    kernel = ConstantKernel(1.0) * Matern(1.0, length_scale_bounds=(0.1, 100.0), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    return GaussianProcessSurrogate(kernel=kernel, n_restarts_optimizer=1)


def ar1():
    return AdditiveCorrectionVFM(lo_surrogate_cls=gp, hi_surrogate_cls=gp, oof_cv=None)


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _data(b, n_lo=12, n_hi=4):
    X = b.sample(n_lo, seed=0)
    return {"lo": (X, b(X, "lo")), "hi": (X[:n_hi], b(X[:n_hi], "hi"))}


def _sims(b):
    return {lvl: (lambda X, lvl=lvl: b(X, lvl)) for lvl in b.levels}


def test_effective_costs():
    costs = {"lo": 1.0, "mid": 3.0, "hi": 10.0}
    assert effective_costs(list(costs), costs, nested=True) == {"lo": 1.0, "mid": 4.0, "hi": 14.0}
    assert effective_costs(list(costs), costs, nested=False) == costs


def test_ar1_variance_split_adds_up():
    b = get_benchmark("forrester")
    model = ar1().fit(_data(b))
    X = b.sample(64, seed=2)
    info = fidelity_info(model, X, b.levels, nested=True)
    np.testing.assert_allclose(info.reduction["hi"], info.std ** 2, rtol=1e-8)
    assert np.all(info.reduction["lo"] <= info.reduction["hi"] + 1e-12)
    assert np.all((info.corr["lo"] >= 0) & (info.corr["lo"] <= 1 + 1e-12))
    not_nested = fidelity_info(model, X, b.levels, nested=False)
    np.testing.assert_allclose(not_nested.reduction["hi"] + not_nested.reduction["lo"], info.std ** 2, rtol=1e-8)


def test_correlation_fallback_for_stacked_model():
    b = get_benchmark("forrester")
    model = StackedVFMSurrogate(["lo", "hi"], surrogate_cls=gp, oof_cv=None).fit(_data(b))
    info = fidelity_info(model, b.sample(64, seed=2), b.levels)
    c = info.corr["lo"][0, 0]
    assert 0 <= c <= 1
    np.testing.assert_allclose(info.reduction["lo"], c ** 2 * info.std ** 2)


def test_level_trust_needs_evidence():
    b = get_benchmark("forrester")
    X = b.sample(16, seed=0)
    model = ar1().fit({"lo": (X, b(X, "hi")), "hi": (X[:3], b(X[:3], "hi"))})
    assert level_trust(model, b.levels, X[:3], b(X[:3]))["lo"][0] == 0.0
    model = ar1().fit({"lo": (X, b(X, "hi")), "hi": (X[:8], b(X[:8], "hi"))})
    assert level_trust(model, b.levels, X[:8], b(X[:8]))["lo"][0] > 0.9
    rng = np.random.default_rng(1)
    noise = ar1().fit({"lo": (X, rng.standard_normal((16, 1))), "hi": (X[:8], b(X[:8], "hi"))})
    assert level_trust(noise, b.levels, X[:8], b(X[:8]))["lo"][0] < 0.5
    info = fidelity_info(model, X, b.levels, trust={"lo": np.zeros(1)})
    assert np.all(info.reduction["lo"] == 0) and np.all(info.corr["lo"] == 0)


def test_cost_accounting_and_budget():
    b = get_benchmark("forrester")
    mf = MultiFidelityActiveLearner(ar1, b.bounds, b.levels, b.costs, n_candidates=128)
    hist = mf.run(_sims(b), budget=60, n_init={"lo": 6, "hi": 3})
    assert mf.total_spent <= 60
    assert mf.spent_["lo"] == len(mf.data_["lo"][0]) * b.costs["lo"]
    assert mf.spent_["hi"] == len(mf.data_["hi"][0]) * b.costs["hi"]
    # Nested: every high-fidelity point was also run at low fidelity.
    lo_rows = {r.tobytes() for r in mf.data_["lo"][0]}
    assert all(r.tobytes() in lo_rows for r in mf.data_["hi"][0])
    assert hist[-1]["cost"] == mf.total_spent
    # Nothing affordable is left (cheapest query costs 1).
    assert 60 - mf.total_spent < 1


def test_min_hi_fraction_is_enforced():
    b = get_benchmark("forrester")
    mf = MultiFidelityActiveLearner(ar1, b.bounds, b.levels, b.costs, min_hi_fraction=0.8, n_candidates=128)
    mf.run(_sims(b), budget=80, n_init={"lo": 6, "hi": 3})
    assert mf.spent_["hi"] / mf.total_spent >= 0.7


def test_uninformative_low_fidelity_is_avoided():
    b = get_benchmark("forrester")
    rng = np.random.default_rng(0)
    sims = {"lo": lambda X: rng.standard_normal((len(X), 1)) * 1e-3, "hi": lambda X: b(X, "hi")}
    mf = MultiFidelityActiveLearner(ar1, b.bounds, b.levels, {"lo": 1.0, "hi": 3.0}, n_candidates=128)
    mf.run(sims, budget=60, n_init={"lo": 6, "hi": 3})
    assert mf.spent_["hi"] / mf.total_spent >= 0.25
    X_hi, Y_hi = mf.data_["hi"]
    assert len(X_hi) >= 6
    assert level_trust(mf.fit(), b.levels, X_hi, Y_hi)["lo"][0] < 0.5


def test_cheap_correlated_low_fidelity_is_preferred():
    b = get_benchmark("forrester")
    sims = {"lo": lambda X: b(X, "hi"), "hi": lambda X: b(X, "hi")}
    mf = MultiFidelityActiveLearner(ar1, b.bounds, b.levels, {"lo": 1.0, "hi": 20.0}, n_candidates=128)
    mf.run(sims, budget=120, n_init={"lo": 6, "hi": 3})
    queried = [h["level"] for h in mf.history_[1:]]
    assert queried.count("lo") > queried.count("hi")


def test_mf_ei_runs_and_validation():
    b = get_benchmark("forrester")
    mf = MultiFidelityActiveLearner(ar1, b.bounds, b.levels, b.costs, acquisition="ei", n_candidates=128)
    mf.run(_sims(b), budget=50, n_init={"lo": 6, "hi": 3})
    assert mf.total_spent <= 50
    with pytest.raises(ValueError):
        MultiFidelityActiveLearner(ar1, b.bounds, b.levels, {"lo": 1.0})
    with pytest.raises(ValueError):
        MultiFidelityActiveLearner(ar1, b.bounds, b.levels, b.costs, acquisition="kg")
    with pytest.raises(ValueError):
        MultiFidelityActiveLearner(ar1, b.bounds, b.levels, b.costs).run(_sims(b), budget=5, n_init={"lo": 6, "hi": 3})
