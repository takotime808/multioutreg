# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from scipy.stats import norm

from multioutreg.metrics import energy_score, gaussian_samples, variogram_score


def _gaussian_crps(y, mu, sigma):
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def test_gaussian_samples_match_requested_covariance():
    cov = np.array([[[1.0, 0.8], [0.8, 2.0]]])
    S = gaussian_samples(np.array([[1.0, -1.0]]), cov, n_samples=20000, seed=0)
    assert S.shape == (1, 20000, 2)
    np.testing.assert_allclose(S[0].mean(axis=0), [1.0, -1.0], atol=0.05)
    np.testing.assert_allclose(np.cov(S[0], rowvar=False), cov[0], atol=0.06)


def test_gaussian_samples_accept_std_and_singular_cov():
    S = gaussian_samples(np.zeros((3, 2)), np.ones((3, 2)), n_samples=10, seed=0)
    assert S.shape == (3, 10, 2)
    singular = np.array([[[1.0, 1.0], [1.0, 1.0]]])
    S = gaussian_samples(np.zeros((1, 2)), singular, n_samples=50, seed=0)
    np.testing.assert_allclose(S[0, :, 0], S[0, :, 1], atol=1e-10)


def test_energy_score_reduces_to_crps_for_one_output():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(20, 1))
    mu, sigma = np.zeros((20, 1)), np.full((20, 1), 1.5)
    S = gaussian_samples(mu, sigma, n_samples=2000, seed=1)
    expected = _gaussian_crps(y[:, 0], 0.0, 1.5)
    np.testing.assert_allclose(energy_score(y, S, average=False), expected, atol=0.04)


def _correlated_truth(n=300, rho=0.9, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    Y = rng.multivariate_normal([0, 0], cov, size=n)
    return Y, np.broadcast_to(cov, (n, 2, 2)), np.broadcast_to(np.eye(2), (n, 2, 2))


def test_scores_prefer_the_true_correlation():
    Y, cov_true, cov_indep = _correlated_truth()
    mean = np.zeros_like(Y)
    S_true = gaussian_samples(mean, cov_true, n_samples=400, seed=1)
    S_indep = gaussian_samples(mean, cov_indep, n_samples=400, seed=1)
    # Same marginals, different dependence: both joint scores tell them apart.
    assert energy_score(Y, S_true) < energy_score(Y, S_indep)
    assert variogram_score(Y, S_true) < 0.5 * variogram_score(Y, S_indep)


def test_variogram_score_weights_and_zero_for_perfect_deterministic_forecast():
    Y = np.array([[1.0, 2.0, 4.0]])
    S = np.repeat(Y[:, None, :], 5, axis=1)
    assert variogram_score(Y, S) == pytest.approx(0.0)
    assert energy_score(Y, S) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        variogram_score(Y, S, weights=np.ones((2, 2)))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        energy_score(np.zeros((3, 2)), np.zeros((3, 10, 3)))
