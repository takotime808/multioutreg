# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from multioutreg.model_selection import AutoDetectMultiOutputRegressor
from multioutreg.surrogates import (
    MultiFidelitySurrogate,
    ConformalPredictionNetworkSurrogate,
)

def test_auto_detect_multi_output_regressor_selects_best():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y_linear = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.randn(200) * 0.01
    y_tree = np.sin(X[:, 0]) + rng.randn(200) * 0.01
    Y = np.column_stack([y_linear, y_tree])

    estimators = [LinearRegression(), DecisionTreeRegressor(random_state=0)]
    param_spaces = [{}, {"max_depth": [1, None]}]
    model = AutoDetectMultiOutputRegressor(estimators, param_spaces)
    model.fit(X, Y)
    preds = model.predict(X)

    assert preds.shape == Y.shape
    assert isinstance(model.models_[0], LinearRegression)
    assert isinstance(model.models_[1], DecisionTreeRegressor)


def test_with_vendored_surrogates_runs():
    rng = np.random.RandomState(1)
    X = rng.rand(100, 3)
    Y = np.column_stack([
        X[:, 0] * 0.5 + rng.randn(100) * 0.01,
        np.sin(X[:, 1]) + rng.randn(100) * 0.01,
    ])

    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=2)
    model.fit(X, Y)
    preds = model.predict(X)

    assert preds.shape == Y.shape


def test_with_vendored_surrogates_includes_conformal_network():
    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=2)
    ctor = model._surrogate_constructors[model._model_names.index("mlp")]
    assert ctor.func is ConformalPredictionNetworkSurrogate


@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=50, n_features=5, n_targets=2, noise=0.1, random_state=42)
    return X, y


def test_init_valid_estimators():
    est = [LinearRegression(), DecisionTreeRegressor()]
    params = [{}, {"max_depth": [1, None]}]
    model = AutoDetectMultiOutputRegressor(est, params)
    assert isinstance(model, AutoDetectMultiOutputRegressor)
    assert model.cv == 3


def test_init_invalid_length():
    est = [LinearRegression()]
    params = [{}, {"max_depth": [1, None]}]
    with pytest.raises(ValueError, match="Each estimator must have a corresponding param space"):
        AutoDetectMultiOutputRegressor(est, params)


def test_fit_and_predict_shape(sample_data):
    X, y = sample_data
    est = [LinearRegression(), DecisionTreeRegressor()]
    params = [{}, {"max_depth": [1, None]}]
    model = AutoDetectMultiOutputRegressor(est, params, cv=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_predict_without_fit_raises():
    est = [LinearRegression()]
    params = [{}]
    model = AutoDetectMultiOutputRegressor(est, params)
    X = np.random.rand(10, 4)
    with pytest.raises(AttributeError, match="Estimator not fitted"):
        model.predict(X)


# def test_with_vendored_surrogates_runs(sample_data):
#     X, y = sample_data
#     model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=2)
#     model.fit(X, y)
#     preds = model.predict(X)
#     assert preds.shape == y.shape


def test_predict_return_std(monkeypatch, sample_data):
    X, y = sample_data
    # Patch the global `return_std` flag in the predict method
    from multioutreg.model_selection import auto_detect

    # Monkeypatch the global flag (simulate toggle behavior)
    monkeypatch.setitem(auto_detect.__dict__, "return_std", True)

    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=2)
    model.fit(X, y)
    pred, std = model.predict(X, return_std=True)
    assert pred.shape == y.shape
    assert std.shape == y.shape


def test_vendored_surrogates_with_multi_fidelity(sample_data):
    X, y = sample_data
    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(
        cv=2, fidelity_levels=["single"]
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert isinstance(model.models_[0], MultiFidelitySurrogate)

def test_multi_output_candidate_scored_out_of_sample():
    # Regression: joint multi-output candidates were scored on training data,
    # so a memorising model (1-NN, zero training error) always beat the
    # cross-validated single-output winners.
    from sklearn.neighbors import KNeighborsRegressor

    rng = np.random.RandomState(0)
    X = rng.rand(150, 3)
    Y = np.column_stack([X @ [1.0, -2.0, 0.5], X @ [0.5, 1.0, -1.0]])
    Y += rng.randn(*Y.shape) * 0.3

    model = AutoDetectMultiOutputRegressor([LinearRegression()], [{}])
    model.register_multi_output_candidates([KNeighborsRegressor(n_neighbors=1)])
    model.fit(X, Y)

    assert all(isinstance(m, LinearRegression) for m in model.models_)


def test_int_cv_uses_shuffled_folds():
    # Regression: cv=int meant unshuffled KFold, so data sorted by an input
    # gave folds that each covered a different input range.
    from sklearn.model_selection import KFold

    splitter = AutoDetectMultiOutputRegressor([LinearRegression()], [{}], cv=3)._splitter()
    assert isinstance(splitter, KFold) and splitter.shuffle


def test_selection_scores_recorded_with_corrected_se(sample_data):
    X, y = sample_data
    model = AutoDetectMultiOutputRegressor(
        [LinearRegression(), DecisionTreeRegressor(random_state=0)], [{}, {}], cv=3
    ).fit(X, y)
    assert len(model.selection_scores_) == y.shape[1]
    scores = model.selection_scores_[0]
    assert set(scores) == {"linear", "dt"}
    assert all(v["se"] > 0 for v in scores.values())


def test_one_se_prefers_simpler_model_within_noise():
    from multioutreg.model_selection.auto_detect import _candidate

    model = AutoDetectMultiOutputRegressor([LinearRegression()], [{}], selection="one_se")
    simple = _candidate("linear", [-1.10, -0.90, -1.00], 0)
    costly = _candidate("gp", [-1.05, -0.85, -0.95], 2)     # better, but within 1 SE
    assert model._choose([simple, costly])["name"] == "linear"
    model.selection = "best"
    assert model._choose([simple, costly])["name"] == "gp"
    far_better = _candidate("gp", [-0.11, -0.09, -0.10], 2)
    model.selection = "one_se"
    assert model._choose([simple, far_better])["name"] == "gp"


def test_invalid_selection_rejected():
    with pytest.raises(ValueError):
        AutoDetectMultiOutputRegressor([LinearRegression()], [{}], selection="1se")


def test_vendored_mlp_scores_the_deployed_conformal_network():
    # Regression: "mlp" was scored as a plain MLPRegressor on 100% of each
    # fold but deployed as ConformalPredictionNetworkSurrogate trained on 80%.
    from multioutreg.surrogates.conformal_network_sklearn import _CPNEstimator

    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
    i = model._model_names.index("mlp")
    assert isinstance(model.estimators[i], _CPNEstimator)
    est = model.estimators[i].fit(*make_regression(60, 3, random_state=0))
    ctor = model._surrogate_constructors[i]
    assert ctor.keywords["random_state"] == est.random_state


def test_vendored_scaled_pipeline_params_reach_surrogate():
    # Pipeline grid keys ("svr__C") must be passed to the surrogate as "C".
    X, y = make_regression(n_samples=60, n_features=3, n_targets=1, noise=0.1,
                           random_state=0)
    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(cv=2)
    keep = [model._model_names.index(n) for n in ("svr", "lasso")]
    model.estimators = [model.estimators[i] for i in keep]
    model.param_spaces = [model.param_spaces[i] for i in keep]
    model._surrogate_constructors = [model._surrogate_constructors[i] for i in keep]
    model._model_names = ["svr", "lasso"]
    model.fit(X, y)
    assert model.models_[0].scale_inputs
    assert model.predict(X).shape == (60, 1)


def test_clone_keeps_vendored_configuration():
    # Regression: clone raised (constructor rewrapped ``estimators``) and would
    # have dropped the surrogate constructors set by with_vendored_surrogates.
    from sklearn.base import clone

    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates(
        cv=2, fidelity_levels=["single"])
    cloned = clone(model)
    assert cloned._model_names == model._model_names
    assert len(cloned._surrogate_constructors) == len(model._surrogate_constructors)
    assert cloned.fidelity_levels == ["single"]

    keep = [model._model_names.index("linear")]
    cloned.estimators = [cloned.estimators[i] for i in keep]
    cloned.param_spaces = [cloned.param_spaces[i] for i in keep]
    cloned._surrogate_constructors = [cloned._surrogate_constructors[i] for i in keep]
    cloned._model_names = ["linear"]
    X, y = make_regression(n_samples=40, n_features=3, n_targets=2, random_state=0)
    cloned.fit(X, y)
    assert isinstance(cloned.models_[0], MultiFidelitySurrogate)


def test_clone_copies_multi_output_candidates_unfitted():
    from sklearn.base import clone
    from sklearn.neighbors import KNeighborsRegressor

    knn = KNeighborsRegressor(n_neighbors=3)
    model = AutoDetectMultiOutputRegressor([LinearRegression()], [{}])
    model.register_multi_output_candidates([knn])
    cloned = clone(model)
    assert len(cloned._multi_output_candidates) == 1
    assert cloned._multi_output_candidates[0] is not knn
    assert cloned._multi_output_candidates[0].get_params() == knn.get_params()
