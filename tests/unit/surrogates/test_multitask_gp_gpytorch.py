# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates import multitask_gp_gpytorch as mt

pytestmark = pytest.mark.skipif(not mt._GPYTORCH_AVAILABLE, reason="gpytorch not installed")


def _correlated_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 1))
    f = np.sin(3 * X[:, 0])
    Y = np.column_stack([f, 2.0 * f + 1.0]) + 0.02 * rng.normal(size=(n, 2))
    return X, Y


def _fit(X, Y, **kw):
    kw = {"max_iter": 100, "random_state": 0, **kw}
    return mt.MultiTaskGPSurrogate(**kw).fit(X, Y)


def test_learns_strong_output_correlation():
    X, Y = _correlated_data()
    model = _fit(X, Y)
    assert model.output_corr_.shape == (2, 2)
    assert model.output_corr_[0, 1] > 0.9


def test_std_matches_cov_diagonal_and_cov_is_psd():
    X, Y = _correlated_data()
    model = _fit(X, Y, predict_chunk=7)  # exercise chunking
    X_test = np.linspace(-1, 1, 15).reshape(-1, 1)
    mean_s, std = model.predict(X_test, return_std=True)
    mean_c, cov = model.predict(X_test, return_cov=True)
    np.testing.assert_allclose(mean_s, mean_c, rtol=1e-6, atol=1e-8)
    assert cov.shape == (15, 2, 2)
    np.testing.assert_allclose(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), std, rtol=1e-4)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8)


def test_predictive_cov_is_correlated_away_from_the_data():
    # Near training points the independent per-output noise dominates; away
    # from them the shared latent function does, so the outputs correlate.
    X, Y = _correlated_data()
    model = _fit(X, Y)
    _, cov = model.predict(np.array([[2.0], [2.5]]), return_cov=True)
    corr = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    assert np.all(corr > 0.5)


def test_missing_outputs_borrow_strength_from_the_other_output():
    # Output 1 is observed only on x < 0; output 0 everywhere. The joint model
    # should still predict output 1 on x > 0 far better than its training mean.
    X, Y = _correlated_data(n=60)
    Y_missing = Y.copy()
    Y_missing[X[:, 0] > 0, 1] = np.nan
    model = _fit(X, Y_missing)
    X_test = np.linspace(0.2, 0.9, 10).reshape(-1, 1)
    truth = 2.0 * np.sin(3 * X_test[:, 0]) + 1.0
    pred = model.predict(X_test)[:, 1]
    mse = np.mean((pred - truth) ** 2)
    mse_mean = np.mean((np.nanmean(Y_missing[:, 1]) - truth) ** 2)
    assert mse < 0.2 * mse_mean


def test_rejects_output_with_no_observations():
    X, Y = _correlated_data(n=10)
    Y[:, 1] = np.nan
    with pytest.raises(ValueError, match="no observed values"):
        _fit(X, Y)


def test_rejects_unknown_kernel():
    with pytest.raises(ValueError):
        mt.MultiTaskGPSurrogate(kernel="linear")


@pytest.mark.parametrize("kernel", ["matern52", "matern32"])
def test_matern_kernels_fit(kernel):
    X, Y = _correlated_data(n=20)
    mean, std = _fit(X, Y, kernel=kernel, max_iter=30).predict(X[:3], return_std=True)
    assert mean.shape == std.shape == (3, 2)
    assert np.all(std > 0)


def test_predict_before_fit_raises():
    with pytest.raises(AttributeError):
        mt.MultiTaskGPSurrogate().predict(np.zeros((2, 1)))
