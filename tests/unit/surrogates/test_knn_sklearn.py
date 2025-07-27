# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.knn_sklearn import KNeighborsSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(0)
    X = np.random.rand(40, 3)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.min(X, axis=1)
    ])
    return X, Y


def test_initialization_default():
    surrogate = KNeighborsSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "KNeighborsRegressor"


def test_initialization_with_params():
    surrogate = KNeighborsSurrogate(n_neighbors=3, weights='distance')
    est = surrogate.model.estimator
    assert est.n_neighbors == 3
    assert est.weights == 'distance'


def test_fit_predict_shape(sample_data):
    X, Y = sample_data
    surrogate = KNeighborsSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_flag(sample_data):
    X, Y = sample_data
    surrogate = KNeighborsSurrogate()
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.allclose(stds, 0.0)


def test_predict_consistency(sample_data):
    X, Y = sample_data
    surrogate = KNeighborsSurrogate()
    surrogate.fit(X, Y)
    pred1 = surrogate.predict(X)
    pred2 = surrogate.predict(X)
    np.testing.assert_allclose(pred1, pred2, rtol=1e-6)
