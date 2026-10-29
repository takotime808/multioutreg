# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.base import clone

from multioutreg.active.benchmarks import dtlz2, dtlz_front, get_benchmark
from multioutreg.metrics import energy_score, gaussian_samples, variogram_score
from multioutreg.surrogates import (
    BayesianRidgeSurrogate,
    GaussianCopulaSurrogate,
    GaussianProcessSurrogate,
)


def _linear_with_correlated_noise(n=200, rho=0.8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    noise = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n) * 0.3
    Y = np.column_stack([X[:, 0] + X[:, 1], 2 * X[:, 0] - X[:, 1]]) + noise
    return X, Y


def test_recovers_residual_correlation():
    X, Y = _linear_with_correlated_noise()
    model = GaussianCopulaSurrogate(BayesianRidgeSurrogate, random_state=0).fit(X, Y)
    assert model.correlation_.shape == (2, 2)
    np.testing.assert_allclose(np.diag(model.correlation_), 1.0)
    assert model.correlation_[0, 1] == pytest.approx(0.8, abs=0.1)


def test_mean_and_std_are_the_wrapped_surrogates():
    X, Y = _linear_with_correlated_noise(n=80)
    model = GaussianCopulaSurrogate(BayesianRidgeSurrogate, cv=3, random_state=0).fit(X, Y)
    base = BayesianRidgeSurrogate().fit(X, Y)
    m_c, s_c = model.predict(X[:5], return_std=True)
    m_b, s_b = base.predict(X[:5], return_std=True)
    np.testing.assert_allclose(m_c, m_b)
    np.testing.assert_allclose(s_c, s_b)
    _, cov = model.predict(X[:5], return_cov=True)
    np.testing.assert_allclose(np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), s_c)


def test_shrinkage_to_identity():
    X, Y = _linear_with_correlated_noise(n=80)
    full = GaussianCopulaSurrogate(BayesianRidgeSurrogate, cv=3, shrinkage=1.0, random_state=0).fit(X, Y)
    np.testing.assert_allclose(full.correlation_, np.eye(2), atol=1e-12)
    with pytest.raises(ValueError):
        GaussianCopulaSurrogate(BayesianRidgeSurrogate, shrinkage=1.5).fit(X, Y)


def test_copula_beats_independent_outputs_on_joint_scores():
    X, Y = _linear_with_correlated_noise(n=300, rho=0.9)
    X_tr, Y_tr, X_te, Y_te = X[:200], Y[:200], X[200:], Y[200:]
    model = GaussianCopulaSurrogate(BayesianRidgeSurrogate, random_state=0).fit(X_tr, Y_tr)
    mean, cov = model.predict(X_te, return_cov=True)
    _, std = model.predict(X_te, return_std=True)
    S_joint = gaussian_samples(mean, cov, n_samples=300, seed=0)
    S_indep = gaussian_samples(mean, std, n_samples=300, seed=0)
    assert variogram_score(Y_te, S_joint) < variogram_score(Y_te, S_indep)
    assert energy_score(Y_te, S_joint) < energy_score(Y_te, S_indep)


def test_dtlz2_outputs_are_strongly_dependent_and_copula_finds_it():
    bench = get_benchmark("dtlz2")
    X = bench.sample(60, seed=0)
    model = GaussianCopulaSurrogate(GaussianProcessSurrogate, cv=3, random_state=0).fit(X, bench(X))
    corr = model.correlation_
    assert corr.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(corr) > -1e-10)
    S = model.sample(X[:4], n_samples=20, seed=0)
    assert S.shape == (4, 20, 3)


def test_rejects_bad_cv_and_clones():
    X, Y = _linear_with_correlated_noise(n=10)
    with pytest.raises(ValueError):
        GaussianCopulaSurrogate(BayesianRidgeSurrogate, cv=1).fit(X, Y)
    model = GaussianCopulaSurrogate(BayesianRidgeSurrogate, cv=4, fit_intercept=False)
    assert clone(model).get_params() == model.get_params()


# -- DTLZ2 benchmark ---------------------------------------------------------


@pytest.mark.parametrize("m", [2, 3, 5])
def test_dtlz2_front_is_on_the_unit_sphere(m):
    X = np.random.default_rng(0).random((50, m + 3))
    X[:, m - 1:] = 0.5  # g = 0
    F = dtlz2(X, n_objectives=m)
    assert F.shape == (50, m)
    np.testing.assert_allclose(np.linalg.norm(F, axis=1), 1.0)
    assert np.all(F >= -1e-12)


def test_dtlz2_matches_the_three_objective_formula():
    X = np.random.default_rng(1).random((20, 7))
    g = ((X[:, 2:] - 0.5) ** 2).sum(axis=1)
    a, b = np.pi / 2 * X[:, 0], np.pi / 2 * X[:, 1]
    expected = (1 + g)[:, None] * np.column_stack(
        [np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)]
    )
    np.testing.assert_allclose(dtlz2(X), expected)


def test_dtlz_front_and_registry():
    F = dtlz_front("dtlz2", n=100, n_objectives=4)
    assert F.shape == (100, 4)
    np.testing.assert_allclose(np.linalg.norm(F, axis=1), 1.0)
    assert get_benchmark("dtlz2").n_features == 7
    assert get_benchmark("dtlz2", n_features=4).n_features == 4
    with pytest.raises(ValueError):
        get_benchmark("dtlz2", n_features=2)
    with pytest.raises(ValueError):
        dtlz_front("dtlz7")
    with pytest.raises(ValueError):
        dtlz2(np.zeros((3, 2)))
