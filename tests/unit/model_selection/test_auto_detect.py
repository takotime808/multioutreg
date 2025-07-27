# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from multioutreg.model_selection import AutoDetectMultiOutputRegressor


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
