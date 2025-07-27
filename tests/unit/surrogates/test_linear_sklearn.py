# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(101)
    X = np.random.rand(60, 5)
    Y = np.column_stack([
        3 * X[:, 0] + 2 * X[:, 1],
        np.mean(X, axis=1)
    ])
    return X, Y


def test_initialization_default():
    surrogate = LinearRegressionSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "LinearRegression"


def test_initialization_with_params():
    surrogate = LinearRegressionSurrogate(fit_intercept=False)
    assert surrogate.model.estimator.fit_intercept is False


def test_fit_and_predict(sample_data):
    X, Y = sample_data
    surrogate = LinearRegressionSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_flag(sample_data):
    X, Y = sample_data
    surrogate = LinearRegressionSurrogate()
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.allclose(stds, 0.0)


def test_prediction_consistency(sample_data):
    X, Y = sample_data
    surrogate = LinearRegressionSurrogate()
    surrogate.fit(X, Y)
    pred1 = surrogate.predict(X)
    pred2 = surrogate.predict(X)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-6)
