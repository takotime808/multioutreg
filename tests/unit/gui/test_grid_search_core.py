# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from multioutreg.gui.grid_search_core import (
    PerTargetRegressorWithStd,
    default_surrogate_defs,
    select_per_target,
)


@pytest.fixture
def two_kinds_of_output():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 2))
    Y = np.column_stack([
        3 * X[:, 0] - X[:, 1],                 # linear
        np.where(X[:, 0] > 0, 5.0, -5.0),      # step
    ]) + 0.05 * rng.normal(size=(200, 2))
    return X, Y


def test_selects_independently_per_output(two_kinds_of_output):
    # Regression: ParameterGrid({"combos": [configs] * n}) iterated the whole
    # config list as one value, so every output always got configs[0].
    X, Y = two_kinds_of_output
    defs = [
        ("dt", DecisionTreeRegressor, {"max_depth": [1]}),
        ("linear", LinearRegression, {}),
    ]
    result = select_per_target(defs, X, Y)
    assert [c["model"] for c in result.combo] == ["linear", "dt"]
    assert isinstance(result.model, PerTargetRegressorWithStd)
    assert result.model.predict(X).shape == Y.shape


def test_joint_model_replaces_winners_only_when_better(two_kinds_of_output):
    X, Y = two_kinds_of_output
    weak = [("dt", DecisionTreeRegressor, {"max_depth": [1]})]
    result = select_per_target(weak, X, Y, moe=KNeighborsRegressor(5))
    assert result.moe_selected
    strong = [("dt", DecisionTreeRegressor, {"max_depth": [1]}),
              ("linear", LinearRegression, {})]
    result = select_per_target(strong, X, Y, moe=KNeighborsRegressor(50))
    assert not result.moe_selected


def test_all_candidates_failing_raises(two_kinds_of_output):
    X, Y = two_kinds_of_output

    def broken(**_):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="Every candidate failed"):
        select_per_target([("broken", broken, {})], X, Y)


@pytest.mark.parametrize("use_moe", [False, True])
def test_on_progress_reports_every_step(two_kinds_of_output, use_moe):
    X, Y = two_kinds_of_output
    defs = [("dt", DecisionTreeRegressor, {"max_depth": [1, 2]}),
            ("linear", LinearRegression, {})]
    calls = []
    select_per_target(defs, X, Y, moe=KNeighborsRegressor(5) if use_moe else None,
                      on_progress=lambda done, total, label: calls.append((done, total, label)))
    total = 3 + use_moe + 1  # three configs, optional MoE, final refit
    assert [c[0] for c in calls] == list(range(total))
    assert all(c[1] == total for c in calls)
    assert calls[0][2] == "Fitting dt {'max_depth': 1}"
    assert calls[-1][2].startswith("Refitting")


def test_every_default_candidate_fits():
    # Regression: "mfs_lr" could never fit (its fit takes one (X, y) tuple).
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=80)
    for name, factory, grid in default_surrogate_defs():
        for params in ParameterGrid(grid):
            pred = factory(**params).fit(X[:60], y[:60]).predict(X[60:])
            assert np.asarray(pred).size == 20, name


@pytest.mark.parametrize("name", ["gpr", "knn", "svr", "elastic_net", "lasso"])
def test_scale_sensitive_candidates_ignore_input_units(name):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    X_mm = X * np.array([1.0, 1000.0])
    factory, grid = next((f, g) for n, f, g in default_surrogate_defs() if n == name)
    params = next(iter(ParameterGrid(grid)))
    a = factory(**params).fit(X, y).predict(X)
    b = factory(**params).fit(X_mm, y).predict(X_mm)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
