# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression

from multioutreg.conformal import (
    JointConformalPredictor,
    SplitConformalPredictor,
    joint_coverage,
)


def _correlated_data(n, n_outputs=3, rho=0.9, seed=0):
    """Linear outputs whose noise is strongly correlated across outputs."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    W = rng.normal(size=(4, n_outputs))
    cov = np.full((n_outputs, n_outputs), rho) + (1 - rho) * np.eye(n_outputs)
    scales = np.arange(1, n_outputs + 1, dtype=float)  # unequal output scales
    noise = rng.multivariate_normal(np.zeros(n_outputs), cov, size=n) * scales
    return X, X @ W + noise


@pytest.fixture(scope="module")
def data():
    X, y = _correlated_data(4000)
    return X[:1000], y[:1000], X[1000:], y[1000:]


@pytest.mark.parametrize("score", ["max", "mahalanobis", "bonferroni"])
def test_joint_coverage_meets_target(data, score):
    X_train, y_train, X_test, y_test = data
    alpha = 0.1
    jcp = JointConformalPredictor(
        LinearRegression(), score=score, calibration_size=0.4, random_state=0
    ).fit(X_train, y_train)
    inside = jcp.contains(X_test, y_test, alpha=alpha)
    assert inside.shape == (len(X_test),)
    assert inside.mean() >= (1 - alpha) - 0.03
    lo, hi = jcp.predict_interval(X_test, alpha=alpha)
    assert lo.shape == hi.shape == y_test.shape
    assert np.all(lo <= hi)
    # The box always contains the region, so it covers at least as often.
    assert joint_coverage(y_test, lo, hi) >= inside.mean()


def test_box_is_the_region_for_max_and_bonferroni(data):
    X_train, y_train, X_test, y_test = data
    for score in ("max", "bonferroni"):
        jcp = JointConformalPredictor(LinearRegression(), score=score, random_state=1)
        jcp.fit(X_train, y_train)
        lo, hi = jcp.predict_interval(X_test, alpha=0.2)
        in_box = np.all((y_test >= lo) & (y_test <= hi), axis=1)
        np.testing.assert_array_equal(in_box, jcp.contains(X_test, y_test, alpha=0.2))


def test_per_output_intervals_undercover_jointly():
    """The gap this module closes: marginal intervals miss jointly."""
    # Five outputs with independent noise, where the union bound bites hardest.
    X, y = _correlated_data(4000, n_outputs=5, rho=0.0, seed=3)
    X, y, Xt, yt = X[:1000], y[:1000], X[1000:], y[1000:]
    alpha = 0.1
    marginal = SplitConformalPredictor(LinearRegression(), random_state=0).fit(X, y)
    lo, hi = marginal.predict_interval(Xt, alpha=alpha)
    assert joint_coverage(yt, lo, hi) < 1 - alpha - 0.1
    joint = JointConformalPredictor(LinearRegression(), random_state=0).fit(X, y)
    lo, hi = joint.predict_interval(Xt, alpha=alpha)
    assert joint_coverage(yt, lo, hi) >= 1 - alpha - 0.03


def test_mahalanobis_smaller_than_box_under_correlation(data):
    X_train, y_train, X_test, _ = data
    vols = {}
    for score in ("max", "mahalanobis", "bonferroni"):
        jcp = JointConformalPredictor(
            LinearRegression(), score=score, calibration_size=0.4, random_state=0
        ).fit(X_train, y_train)
        vols[score] = float(np.mean(jcp.log_volume(X_test[:5], alpha=0.1)))
    assert vols["mahalanobis"] < vols["max"]
    assert vols["mahalanobis"] < vols["bonferroni"]


def test_adaptive_uses_model_std():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(300, 1))
    # Noise grows with x, so adaptive regions should widen with x.
    y = np.column_stack([np.sin(6 * X[:, 0]), np.cos(6 * X[:, 0])])
    y = y + rng.normal(size=y.shape) * (0.05 + 0.3 * X)
    gp = GaussianProcessRegressor(RBF(0.3) + WhiteKernel(0.05), normalize_y=True)
    jcp = JointConformalPredictor(gp, adaptive=True, random_state=0).fit(X, y)
    lo, hi = jcp.predict_interval(np.array([[0.1], [0.9]]), alpha=0.1)
    assert lo.shape == (2, 2)
    assert np.all(np.isfinite(hi - lo))


def test_single_output_matches_split_conformal(data):
    X_train, y_train, X_test, _ = data
    split = SplitConformalPredictor(LinearRegression(), random_state=0)
    joint = JointConformalPredictor(LinearRegression(), score="bonferroni", random_state=0)
    lo_s, hi_s = split.fit(X_train, y_train[:, 0]).predict_interval(X_test, 0.1)
    lo_j, hi_j = joint.fit(X_train, y_train[:, 0]).predict_interval(X_test, 0.1)
    assert lo_j.shape == (len(X_test),)
    np.testing.assert_allclose(lo_s, lo_j)
    np.testing.assert_allclose(hi_s, hi_j)


def test_shape_from_training_residuals(data):
    X_train, y_train, X_test, y_test = data
    jcp = JointConformalPredictor(
        LinearRegression(), score="mahalanobis", shape_size=0, random_state=0
    ).fit(X_train, y_train)
    assert len(jcp.scores_) == 200  # every calibration row sets the quantile
    assert jcp.contains(X_test, y_test, 0.1).mean() >= 0.87


def test_singular_covariance_is_regularized():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y1 = X[:, 0] + rng.normal(size=200)
    y = np.column_stack([y1, 2 * y1])  # perfectly collinear outputs
    jcp = JointConformalPredictor(LinearRegression(), score="mahalanobis", random_state=0)
    jcp.fit(X, y)
    assert np.all(np.isfinite(jcp.precision_))
    lo, hi = jcp.predict_interval(X[:3], 0.1)
    assert np.all(np.isfinite(hi - lo))


def test_too_few_calibration_points_gives_infinite_region(data):
    X_train, y_train, X_test, _ = data
    jcp = JointConformalPredictor(LinearRegression(), calibration_size=0.01, random_state=0)
    jcp.fit(X_train, y_train)  # 10 calibration rows, 5 for the quantile
    with pytest.warns(UserWarning, match="too few"):
        lo, hi = jcp.predict_interval(X_test, alpha=0.1)
    assert np.all(np.isinf(hi))
    with pytest.warns(UserWarning):
        assert np.all(np.isinf(jcp.log_volume(X_test, alpha=0.1)))


@pytest.mark.parametrize(
    "kwargs, match",
    [({"score": "l2"}, "score"), ({"shape_size": 1.0}, "shape_size")],
)
def test_bad_parameters(data, kwargs, match):
    X_train, y_train, _, _ = data
    with pytest.raises(ValueError, match=match):
        JointConformalPredictor(LinearRegression(), **kwargs).fit(X_train, y_train)


def test_unfitted_raises():
    jcp = JointConformalPredictor(LinearRegression())
    for call in (jcp.predict_interval, jcp.log_volume):
        with pytest.raises(AttributeError, match="not fitted"):
            call(np.zeros((1, 2)))


def test_joint_coverage_metric():
    y = np.array([[0.0, 0.0], [1.0, 5.0], [2.0, 2.0]])
    lo = np.zeros_like(y)
    hi = np.full_like(y, 3.0)
    assert joint_coverage(y, lo, hi) == pytest.approx(2 / 3)
    assert joint_coverage(y[:, 0], lo[:, 0], hi[:, 0]) == 1.0
