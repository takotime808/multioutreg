# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from multioutreg.surrogates.base_sklearn import BaseSurrogate


@pytest.fixture
def sample_data():
    # Create a simple 2D input and 2-output target
    X = np.random.rand(20, 5)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.prod(X, axis=1)
    ])
    return X, Y


def test_initialization_with_linear_model():
    base = BaseSurrogate(LinearRegression())
    assert hasattr(base, "model")
    assert isinstance(base.model.estimator, LinearRegression)


def test_fit_and_predict(sample_data):
    X, Y = sample_data
    base = BaseSurrogate(LinearRegression())
    base.fit(X, Y)
    preds = base.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_supported_model(sample_data):
    # GaussianProcessRegressor supports return_std=True
    X, Y = sample_data
    gpr = BaseSurrogate(GaussianProcessRegressor())
    gpr.fit(X, Y)
    preds, stds = gpr.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert isinstance(stds, np.ndarray)


def test_predict_with_return_std_unsupported_model(sample_data):
    # LinearRegression does not support return_std=True
    X, Y = sample_data
    base = BaseSurrogate(LinearRegression())
    base.fit(X, Y)
    preds, stds = base.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.allclose(stds, 0)


def test_invalid_input_shape():
    base = BaseSurrogate(LinearRegression())
    X = np.random.rand(10, 3)
    Y = np.random.rand(10, 2)
    base.fit(X, Y)
    with pytest.raises(ValueError):
        # Incorrect input dimension
        base.predict(np.random.rand(5, 2, 2))

