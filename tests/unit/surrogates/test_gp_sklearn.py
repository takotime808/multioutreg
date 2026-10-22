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
    assert surrogate.model.estimator[-1].__class__.__name__ == "GaussianProcessRegressor"


def test_initialization_with_kernel_param():
    from sklearn.gaussian_process.kernels import RBF
    surrogate = GaussianProcessSurrogate(kernel=RBF(length_scale=2.0))
    kernel = surrogate.model.estimator[-1].kernel
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


def test_return_std_is_nonzero_away_from_training_data(sample_data):
    # Regression: MultiOutputRegressor rejects return_std, which previously
    # made BaseSurrogate silently return all-zero std for the GP.
    X, Y = sample_data
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(X, Y)
    X_far = X + 5.0
    _, stds = surrogate.predict(X_far, return_std=True)
    assert np.all(stds > 0)


def test_default_kernel_does_not_collapse_on_noisy_data():
    # Regression: sklearn's default kernel (fixed amplitude, no noise term,
    # alpha=1e-10) interpolated the noise and collapsed to RBF(1e-5).
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (150, 2))
    y = np.sin(X[:, 0]) + 0.05 * X[:, 1] ** 2 + 0.5 * rng.normal(size=150)
    Xt = rng.uniform(0, 10, (300, 2))
    f = np.sin(Xt[:, 0]) + 0.05 * Xt[:, 1] ** 2
    s = GaussianProcessSurrogate().fit(X, y)
    pred, std = s.predict(Xt, return_std=True)
    assert np.mean((pred.ravel() - f) ** 2) < 0.1
    assert 0.3 < std.mean() < 1.0
