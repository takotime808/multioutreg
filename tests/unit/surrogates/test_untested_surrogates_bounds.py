# Copyright (c) 2026 takotime808
"""Shape, bounds and failure-point tests for surrogates without a dedicated test file."""

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from multioutreg.surrogates import (
    ARDGPSurrogate,
    BayesianRidgeSurrogate,
    HistGradientBoostingSurrogate,
    NystroemGPSurrogate,
    PolynomialBayesianRidgeSurrogate,
    RFFGPSurrogate,
)
from multioutreg.surrogates.forest_uncertainty import (
    ForestSurrogate,
    estimate_noise_variance,
    tree_predictions,
)
from multioutreg.surrogates.rfgp_sklearn import _RFFEstimator


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(60, 2))
    Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] * X[:, 1]])
    Y = Y + 0.05 * rng.normal(size=Y.shape)
    return X, Y


UQ_FACTORIES = {
    "ard_gp": lambda: ARDGPSurrogate(),
    "rff_rbf": lambda: RFFGPSurrogate(n_components=100, length_scale=0.5, random_state=0),
    "rff_matern32": lambda: RFFGPSurrogate(n_components=100, kernel="matern32", random_state=0),
    "rff_matern52": lambda: RFFGPSurrogate(n_components=100, kernel="matern52", random_state=0),
    "nystroem": lambda: NystroemGPSurrogate(n_components=30, random_state=0),
    "bayes_ridge": lambda: BayesianRidgeSurrogate(),
    "poly_bayes_ridge": lambda: PolynomialBayesianRidgeSurrogate(degree=2),
    "forest": lambda: ForestSurrogate(RandomForestRegressor(n_estimators=20, random_state=0)),
}


@pytest.mark.parametrize("name", UQ_FACTORIES)
def test_uq_surrogate_shapes_and_positive_std(name, data):
    X, Y = data
    m = UQ_FACTORIES[name]().fit(X, Y)
    mean = m.predict(X[:7])
    mean2, std = m.predict(X[:7], return_std=True)
    assert mean.shape == mean2.shape == std.shape == (7, 2)
    np.testing.assert_allclose(mean, mean2)
    assert np.all(np.isfinite(std)) and np.all(std >= 0)


@pytest.mark.parametrize("name", UQ_FACTORIES)
def test_uq_surrogate_conformal_intervals_contain_point_prediction(name, data):
    X, Y = data
    m = UQ_FACTORIES[name]().fit(X[:40], Y[:40])
    with pytest.raises(AttributeError, match="wrap_conformal"):
        m.conformal_predict(X[:3])
    m.wrap_conformal(X[40:], Y[40:])
    lo, hi = m.conformal_predict(X[:5], alpha=0.5)
    pred = np.asarray(m.predict(X[:5])).reshape(5, -1)
    assert lo.shape == hi.shape == (5, 2)
    assert np.all(lo <= pred) and np.all(pred <= hi)


@pytest.mark.parametrize("name", ["ard_gp", "rff_rbf", "nystroem"])
def test_uq_surrogate_accepts_1d_target(name, data):
    X, Y = data
    m = UQ_FACTORIES[name]().fit(X, Y[:, 0])
    assert np.asarray(m.predict(X[:4])).reshape(4, -1).shape == (4, 1)


@pytest.mark.parametrize("name", ["bayes_ridge", "poly_bayes_ridge"])
def test_base_surrogates_treat_1d_target_as_one_output(name, data):
    # Regression: BaseSurrogate passed 1-D Y straight to MultiOutputRegressor,
    # which rejects it, unlike the per-output ConformalMixin surrogates above.
    X, Y = data
    m = UQ_FACTORIES[name]().fit(X, Y[:, 0])
    mean, std = m.predict(X[:4], return_std=True)
    assert mean.shape == std.shape == (4, 1)


def test_hist_gb_accepts_1d_target(data):
    X, Y = data
    m = HistGradientBoostingSurrogate(max_iter=10, min_samples_leaf=5).fit(X, Y[:, 0])
    assert m.predict(X[:3]).shape == (3, 1)


@pytest.mark.parametrize("name", ["ard_gp", "rff_rbf", "nystroem", "bayes_ridge"])
def test_uq_surrogate_predict_before_fit_raises(name):
    with pytest.raises(Exception):
        UQ_FACTORIES[name]().predict(np.zeros((2, 2)))


def test_ard_gp_std_grows_away_from_training_data(data):
    X, Y = data
    m = ARDGPSurrogate().fit(X, Y)
    _, s_in = m.predict(X[:5], return_std=True)
    _, s_out = m.predict(np.full((5, 2), 10.0), return_std=True)
    assert np.all(s_out.mean(axis=0) > s_in.mean(axis=0))


# ---------------------------------------------------------------------------
# RFF-GP
# ---------------------------------------------------------------------------


def test_rff_unknown_kernel_raises(data):
    X, Y = data
    with pytest.raises(ValueError, match="Unknown kernel"):
        RFFGPSurrogate(kernel="laplace").fit(X, Y)


def test_rff_is_reproducible_and_seed_sensitive(data):
    X, Y = data
    a = RFFGPSurrogate(n_components=50, random_state=1).fit(X, Y).predict(X[:5])
    b = RFFGPSurrogate(n_components=50, random_state=1).fit(X, Y).predict(X[:5])
    c = RFFGPSurrogate(n_components=50, random_state=2).fit(X, Y).predict(X[:5])
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_rff_features_are_bounded_by_sqrt_2_over_D(data):
    X, Y = data
    est = _RFFEstimator(n_components=64, random_state=0).fit(X, Y[:, 0])
    Z = est._rff_transform(X)
    assert Z.shape == (len(X), 64)
    assert np.all(np.abs(Z) <= np.sqrt(2.0 / 64) + 1e-12)


def test_rff_kernel_approximation_matches_rbf_on_average():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 2))
    # Sample the features directly: fitting BayesianRidge on 20k columns is slow.
    D = 20000
    est = _RFFEstimator(n_components=D, length_scale=1.0)
    est.omega_ = est._sample_frequencies(rng, X.shape[1], D)
    est.bias_ = rng.uniform(0.0, 2.0 * np.pi, size=D)
    Z = est._rff_transform(X)
    K_approx = Z @ Z.T
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    np.testing.assert_allclose(K_approx, np.exp(-d2 / 2), atol=0.05)


# ---------------------------------------------------------------------------
# Nystroem
# ---------------------------------------------------------------------------


def test_nystroem_more_landmarks_than_samples_is_clamped(data):
    X, Y = data
    m = NystroemGPSurrogate(n_components=10_000, random_state=0).fit(X[:15], Y[:15])
    assert all(e._nystroem.n_components == 15 for e in m.estimators_)


# ---------------------------------------------------------------------------
# Polynomial Bayesian ridge
# ---------------------------------------------------------------------------


def test_poly_bayes_ridge_recovers_exact_quadratic():
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(80, 2))
    Y = (1 + 2 * X[:, 0] - X[:, 1] + 3 * X[:, 0] * X[:, 1] + X[:, 1] ** 2)[:, None]
    m = PolynomialBayesianRidgeSurrogate(degree=2).fit(X, Y)
    np.testing.assert_allclose(m.predict(X[:5]).ravel(), Y[:5, 0], atol=1e-3)


def test_poly_bayes_ridge_interaction_only_has_fewer_features(data):
    X, Y = data
    full = PolynomialBayesianRidgeSurrogate(degree=2).fit(X, Y)
    inter = PolynomialBayesianRidgeSurrogate(degree=2, interaction_only=True).fit(X, Y)
    n_full = full.model.estimators_[0]._poly.n_output_features_
    n_inter = inter.model.estimators_[0]._poly.n_output_features_
    assert (n_full, n_inter) == (5, 3)


# ---------------------------------------------------------------------------
# HistGradientBoosting
# ---------------------------------------------------------------------------


def test_hist_gb_return_std_falls_back_to_zero(data):
    X, Y = data
    m = HistGradientBoostingSurrogate(max_iter=20, min_samples_leaf=5).fit(X, Y)
    pred, std = m.predict(X[:4], return_std=True)
    assert pred.shape == (4, 2)
    np.testing.assert_array_equal(std, 0.0)


def test_hist_gb_get_and_set_params_round_trip(data):
    m = HistGradientBoostingSurrogate(max_iter=10, l2_regularization=0.5)
    p = m.get_params()
    assert p["max_iter"] == 10 and p["l2_regularization"] == 0.5
    m.set_params(max_iter=7, l2_regularization=0.1)
    est = m.model.estimator
    assert est.max_iter == 7 and est.l2_regularization == 0.1
    assert m.get_params()["l2_regularization"] == 0.1


def test_hist_gb_invalid_learning_rate_raises_on_fit(data):
    X, Y = data
    with pytest.raises(ValueError):
        HistGradientBoostingSurrogate(learning_rate=-1.0).fit(X, Y)


# ---------------------------------------------------------------------------
# Forest noise estimation
# ---------------------------------------------------------------------------


def test_tree_predictions_shape(data):
    X, Y = data
    rf = RandomForestRegressor(n_estimators=7, random_state=0).fit(X, Y[:, 0])
    assert tree_predictions(rf, X[:3]).shape == (7, 3)


def test_noise_variance_is_zero_for_fewer_than_four_points(data):
    X, Y = data
    rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(X[:3], Y[:3, 0])
    assert estimate_noise_variance(rf, X[:3], Y[:3, 0]) == 0.0
    assert estimate_noise_variance(rf, X[:3], Y[:3]).shape == (2,)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_noise_variance_is_non_negative_and_tracks_noise_level(bootstrap):
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(150, 1))
    clean = np.sin(6 * X[:, 0])
    vals = []
    for sigma in (0.01, 0.5):
        y = clean + sigma * rng.normal(size=150)
        Forest = RandomForestRegressor if bootstrap else ExtraTreesRegressor
        f = Forest(n_estimators=30, bootstrap=bootstrap, random_state=0, min_samples_leaf=3).fit(X, y)
        vals.append(estimate_noise_variance(f, X, y, cv=3))
    assert all(v >= 0 for v in vals)
    assert vals[1] > vals[0]


def test_noise_variance_falls_back_to_kfold_when_too_few_oob_points():
    rng = np.random.default_rng(4)
    X = rng.uniform(size=(20, 1))
    y = X[:, 0] + 0.1 * rng.normal(size=20)
    # Two trees: almost no point is out-of-bag for >= 2 trees.
    f = RandomForestRegressor(n_estimators=2, random_state=0).fit(X, y)
    assert estimate_noise_variance(f, X, y, cv=3) >= 0.0


def test_forest_surrogate_without_noise_has_smaller_std(data):
    X, Y = data
    rf = lambda: RandomForestRegressor(n_estimators=20, random_state=0)  # noqa: E731
    with_noise = ForestSurrogate(rf()).fit(X, Y)
    without = ForestSurrogate(rf(), include_noise=False).fit(X, Y)
    np.testing.assert_array_equal(without.noise_var_, 0.0)
    _, s1 = with_noise.predict(X[:10], return_std=True)
    _, s0 = without.predict(X[:10], return_std=True)
    assert np.all(s1 >= s0 - 1e-12)
