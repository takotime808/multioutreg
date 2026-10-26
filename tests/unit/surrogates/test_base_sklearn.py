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



def test_return_std_matches_underlying_estimators(sample_data):
    X, Y = sample_data
    base = BaseSurrogate(GaussianProcessRegressor())
    base.fit(X, Y)
    preds, stds = base.predict(X + 1.0, return_std=True)
    for j, est in enumerate(base.model.estimators_):
        p, s = est.predict(X + 1.0, return_std=True)
        np.testing.assert_allclose(preds[:, j], p)
        np.testing.assert_allclose(stds[:, j], s)


@pytest.mark.parametrize("name", ["SVRSurrogate", "KNeighborsSurrogate",
                                  "GaussianProcessSurrogate", "LassoSurrogate",
                                  "ElasticNetSurrogate"])
def test_scale_sensitive_surrogates_ignore_input_units(name):
    # Regression: unscaled inputs made distance/kernel/penalty models depend
    # on units (e.g. one input in mm, another in m).
    import multioutreg.surrogates as sur

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 2))
    y = (np.sin(3 * X[:, 0]) + X[:, 1])[:, None]
    X_mm = X * np.array([1.0, 1000.0])
    a = getattr(sur, name)().fit(X, y).predict(X)
    b = getattr(sur, name)().fit(X_mm, y).predict(X_mm)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


class _KwargsSwallowingRegressor(LinearRegression):
    """Mimics LGBMRegressor: ``predict`` accepts ``**kwargs`` but returns only the mean."""

    def predict(self, X, **kwargs):
        return super().predict(X)


@pytest.mark.parametrize("n_rows", [2, 7])
def test_return_std_with_estimator_that_ignores_return_std(sample_data, n_rows):
    # Regression: ``p, s = est.predict(X, return_std=True)`` raised for most
    # sizes and silently split the mean array into (mean, std) for 2 rows.
    X, Y = sample_data
    base = BaseSurrogate(_KwargsSwallowingRegressor()).fit(X, Y)
    mean, std = base.predict(X[:n_rows], return_std=True)
    np.testing.assert_allclose(mean, base.predict(X[:n_rows]))
    assert std.shape == (n_rows, Y.shape[1])
    assert np.all(std == 0)
