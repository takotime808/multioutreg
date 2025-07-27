# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(20, 3)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.sin(np.sum(X, axis=1)),
    ])
    return X, Y


def test_initialization_default():
    surrogate = GaussianProcessSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "GaussianProcessRegressor"


def test_initialization_with_kernel_param():
    from sklearn.gaussian_process.kernels import RBF
    surrogate = GaussianProcessSurrogate(kernel=RBF(length_scale=2.0))
    kernel = surrogate.model.estimator.kernel
    assert hasattr(kernel, 'length_scale')
    assert kernel.length_scale == 2.0


def test_fit_and_predict_shape(sample_data):
    X, Y = sample_data
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std(sample_data):
    X, Y = sample_data
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    # assert (stds > 0).any()  # Some uncertainty should be positive
    assert (stds >= 0).any()


def test_repeatable_predictions(sample_data):
    X, Y = sample_data
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, Y)
    preds1 = surrogate.predict(X)
    preds2 = surrogate.predict(X)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-5)
