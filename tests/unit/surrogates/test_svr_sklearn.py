# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.svr_sklearn import SVRSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(30, 3)
    Y = np.column_stack([
        np.sin(X[:, 0] * 2 * np.pi),
        np.log(X[:, 1] + 1)
    ])
    return X, Y


def test_initialization_default():
    surrogate = SVRSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "SVR"


def test_initialization_with_params():
    surrogate = SVRSurrogate(C=2.0, kernel='rbf')
    est = surrogate.model.estimator
    assert est.C == 2.0
    assert est.kernel == 'rbf'


def test_fit_predict_output(sample_data):
    X, Y = sample_data
    surrogate = SVRSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_flag(sample_data):
    X, Y = sample_data
    surrogate = SVRSurrogate()
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.allclose(stds, 0.0)


def test_prediction_consistency(sample_data):
    X, Y = sample_data
    surrogate = SVRSurrogate()
    surrogate.fit(X, Y)
    pred1 = surrogate.predict(X)
    pred2 = surrogate.predict(X)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-6)
