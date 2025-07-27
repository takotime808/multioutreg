# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.gradient_boosting_sklearn import GradientBoostingSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(123)
    X = np.random.rand(50, 4)
    Y = np.column_stack([
        X.sum(axis=1),
        np.sqrt(X[:, 0] + 1),
    ])
    return X, Y


def test_initialization_default():
    surrogate = GradientBoostingSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "GradientBoostingRegressor"


def test_initialization_with_params():
    surrogate = GradientBoostingSurrogate(n_estimators=50, learning_rate=0.1)
    est = surrogate.model.estimator
    assert est.n_estimators == 50
    assert est.learning_rate == 0.1


def test_fit_predict(sample_data):
    X, Y = sample_data
    surrogate = GradientBoostingSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_flag(sample_data):
    X, Y = sample_data
    surrogate = GradientBoostingSurrogate()
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.allclose(stds, 0.0)


def test_consistency(sample_data):
    X, Y = sample_data
    surrogate = GradientBoostingSurrogate()
    surrogate.fit(X, Y)
    pred1 = surrogate.predict(X)
    pred2 = surrogate.predict(X)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-6)
