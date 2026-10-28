# Copyright (c) 2026 takotime808

import io
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from multioutreg.gui.grid_search_core import (
    PerTargetRegressorWithStd,
    build_model_bundle,
    build_model_pipeline,
    default_surrogate_defs,
    select_per_target,
    serialize_model_bundle,
)
from multioutreg.utils.imputation import impute_train_test
from multioutreg.utils.pca_utils import fit_input_pca


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


def test_model_pipeline_replays_app_preprocessing(two_kinds_of_output):
    X, Y = two_kinds_of_output
    X = np.column_stack([X, X[:, 0] + X[:, 1]])
    X[::17, 1] = np.nan
    X_tr, X_te, Y_tr = X[:150], X[150:], Y[:150]
    # Same steps, in the same order, as the app.
    X_tr_i, X_te_i, imputer = impute_train_test(X_tr, X_te, return_imputer=True)
    pca = fit_input_pca(X_tr_i, X_te_i, "Manual", 2)
    model = select_per_target(
        [("linear", LinearRegression, {}), ("dt", DecisionTreeRegressor, {"max_depth": [3]})],
        pca.X_train, Y_tr).model
    expected = model.predict(pca.X_test)

    cols = ["a", "b", "c"]
    pipe = build_model_pipeline(model, cols, imputer, pca.transformer)
    assert [name for name, _ in pipe.steps] == ["columns", "imputer", "pca", "model"]
    # A DataFrame with shuffled and extra columns is reordered by name.
    df = pd.DataFrame(X_te, columns=cols)[["c", "a", "b"]].assign(extra=1.0)
    np.testing.assert_allclose(pipe.predict(df), expected)
    mean, std = pipe.predict(X_te, return_std=True)
    np.testing.assert_allclose(mean, expected)
    assert std.shape == expected.shape
    with pytest.raises(KeyError):
        pipe.predict(df.drop(columns="b"))

    bundle = build_model_bundle(
        pipe, cols, ["y0", "y1"], [{"model": "linear"}, {"model": "dt", "max_depth": 3}],
        pca_settings={"method": "Manual", "n_components": 2},
        conformal_alpha=0.1, conformal_quantiles=np.array([0.5, 1.5]))
    loaded = joblib.load(io.BytesIO(serialize_model_bundle(bundle)))
    np.testing.assert_allclose(loaded["pipeline"].predict(df), expected)
    assert loaded["conformal"] == {"alpha": 0.1, "quantiles": {"y0": 0.5, "y1": 1.5}}
    config = loaded["config"]
    assert json.loads(json.dumps(config)) == config
    assert config["preprocessing"]["imputer"] == {"type": "KNNImputer", "n_neighbors": 5}
    assert config["preprocessing"]["pca"]["n_components"] == 2
    assert config["versions"]["scikit-learn"]


def test_model_bundle_config_is_json_safe_with_kernel_params():
    pipe = build_model_pipeline(LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0]), ["x"])
    assert [name for name, _ in pipe.steps] == ["columns", "model"]
    bundle = build_model_bundle(pipe, ["x"], ["y"], [{"model": "gpr", "kernel": RBF(2.0)}])
    assert bundle["config"]["models"][0]["kernel"] == repr(RBF(2.0))
    assert bundle["config"]["preprocessing"] == {"imputer": None, "pca": None}
    assert bundle["conformal"] is None


def test_selection_is_cross_validated_by_default(two_kinds_of_output):
    X, Y = two_kinds_of_output
    defs = [("linear", LinearRegression, {})]
    result = select_per_target(defs, X, Y)
    assert result.cv_description == "10-Fold CV (shuffled)"
    small = select_per_target(defs, X[:40], Y[:40])
    assert small.cv_description == "Repeated 5-Fold CV (x3 repeats)"
    legacy = select_per_target(defs, X, Y, val_fraction=0.2)
    assert legacy.cv_description == "a single 20% validation split"


def test_per_target_grid_search_estimator_refits_selection(two_kinds_of_output):
    from multioutreg.conformal.cv_plus import CVPlusConformalPredictor
    from multioutreg.gui.grid_search_core import PerTargetGridSearch

    X, Y = two_kinds_of_output
    defs = [("dt", DecisionTreeRegressor, {"max_depth": [1]}), ("linear", LinearRegression, {})]
    search = PerTargetGridSearch(defs, moe_factory=lambda: KNeighborsRegressor(50))
    pred, std = search.fit(X, Y).predict(X[:5], return_std=True)
    assert pred.shape == std.shape == (5, 2)
    assert [c["model"] for c in search.result_.combo] == ["linear", "dt"]
    assert search.skipped_ == []
    # CV+ can refit the whole selection per fold (it deep-copies the wrapper).
    cvp = CVPlusConformalPredictor(search, n_folds=3, random_state=0).fit(X[:60], Y[:60])
    lo, hi = cvp.predict_interval(X[60:65], alpha=0.2)
    assert lo.shape == (5, 2) and np.all(lo < hi)
