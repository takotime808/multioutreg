# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.base import clone

from multioutreg.surrogates import (
    BayesianRidgeSurrogate,
    LinearRegressionSurrogate,
    PCAOutputSurrogate,
)


def _field_data(n=60, m=40, seed=0):
    """Outputs on a 1-D grid driven by two latent factors, so rank(Y) ~ 2."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    t = np.linspace(0, 1, m)
    Y = np.outer(X[:, 0], np.sin(2 * np.pi * t)) + np.outer(X[:, 1] ** 2, t)
    return X, Y + 0.01 * rng.normal(size=Y.shape)


def test_all_components_reconstruct_linear_outputs_exactly():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    Y = X @ rng.normal(size=(3, 4)) + 2.0
    model = PCAOutputSurrogate(LinearRegressionSurrogate, n_components=4).fit(X, Y)
    assert model.n_components_ == 4
    np.testing.assert_allclose(model.predict(X), Y, atol=1e-8)
    np.testing.assert_allclose(model.truncation_var_, 0.0, atol=1e-12)


def test_reduces_field_output_to_few_components():
    X, Y = _field_data()
    model = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=0.99).fit(X[:, :1], Y)
    assert model.n_components_ <= 3
    assert model.explained_variance_ratio_.sum() >= 0.99
    assert model.predict(X[:5, :1]).shape == (5, Y.shape[1])


def test_variance_fraction_one_keeps_every_component():
    X, Y = _field_data(n=20, m=6)
    model = PCAOutputSurrogate(LinearRegressionSurrogate, n_components=1.0).fit(X, Y)
    assert model.n_components_ == 6


@pytest.mark.parametrize("bad", [0, -1, 0.0, 1.5])
def test_rejects_bad_n_components(bad):
    X, Y = _field_data(n=10, m=4)
    with pytest.raises(ValueError):
        PCAOutputSurrogate(LinearRegressionSurrogate, n_components=bad).fit(X, Y)


def test_std_is_sqrt_of_cov_diagonal_and_cov_is_psd():
    X, Y = _field_data(n=40, m=8)
    model = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=2).fit(X, Y)
    mean_s, std = model.predict(X[:7], return_std=True)
    mean_c, cov = model.predict(X[:7], return_cov=True)
    np.testing.assert_allclose(mean_s, mean_c)
    assert cov.shape == (7, 8, 8)
    np.testing.assert_allclose(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), std, rtol=1e-10)
    np.testing.assert_allclose(cov, np.transpose(cov, (0, 2, 1)))
    assert np.all(np.linalg.eigvalsh(cov) > -1e-10)


def test_truncation_error_widens_std():
    X, Y = _field_data(n=40, m=8)
    with_err = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=1).fit(X, Y)
    without = PCAOutputSurrogate(
        BayesianRidgeSurrogate, n_components=1, include_truncation_error=False
    ).fit(X, Y)
    assert np.all(with_err.truncation_var_ > 0)
    _, s_with = with_err.predict(X[:5], return_std=True)
    _, s_without = without.predict(X[:5], return_std=True)
    assert np.all(s_with > s_without)


def test_one_component_makes_outputs_fully_correlated():
    # Two outputs that are scaled copies of each other load on one component.
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(40, 1))
    base = np.sin(3 * X[:, 0])
    Y = np.column_stack([base, -2.0 * base])
    model = PCAOutputSurrogate(
        BayesianRidgeSurrogate, n_components=1, include_truncation_error=False
    ).fit(X, Y)
    _, cov = model.predict(X[:3], return_cov=True)
    corr = cov[:, 0, 1] / np.sqrt(cov[:, 0, 0] * cov[:, 1, 1])
    np.testing.assert_allclose(corr, -1.0, atol=1e-8)


def test_standardize_off_keeps_unit_scale():
    X, Y = _field_data(n=20, m=4)
    model = PCAOutputSurrogate(LinearRegressionSurrogate, standardize=False).fit(X, Y)
    np.testing.assert_array_equal(model.y_scale_, 1.0)


def test_surrogate_kwargs_and_clone():
    model = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=2, fit_intercept=False)
    params = model.get_params()
    assert params["n_components"] == 2 and params["fit_intercept"] is False
    fresh = clone(model)
    assert fresh is not model and fresh.get_params() == params


def test_predict_components_shape():
    X, Y = _field_data(n=30, m=10)
    model = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=3).fit(X, Y)
    z, s = model.predict_components(X[:4], return_std=True)
    assert z.shape == s.shape == (4, 3)
