# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from multioutreg.conformal.cv_plus import CVPlusConformalPredictor


@pytest.fixture
def single_output_data():
    X, y = make_regression(n_samples=300, n_features=5, noise=10, random_state=42)
    return X[:200], y[:200], X[200:], y[200:]


@pytest.fixture
def multi_output_data():
    X, y = make_regression(
        n_samples=300, n_features=5, n_targets=3, noise=10, random_state=42
    )
    return X[:200], y[:200], X[200:], y[200:]


class TestCVPlusSingleOutput:
    def test_fit_and_predict_interval(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100,)
        assert y_upper.shape == (100,)
        assert np.all(y_lower <= y_upper)

    def test_coverage_guarantee(self, single_output_data):
        X_train, y_train, X_test, y_test = single_output_data
        alpha = 0.1
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        coverage = np.mean((y_test >= y_lower) & (y_test <= y_upper))
        assert coverage >= (1 - alpha) - 0.1

    def test_predict_without_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        y_pred = cp.predict(X_test)
        assert y_pred.shape == (100,)

    def test_predict_with_alpha(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), random_state=0)
        cp.fit(X_train, y_train)
        result = cp.predict(X_test, alpha=0.1)
        assert len(result) == 3
        y_pred, y_lower, y_upper = result
        assert y_pred.shape == (100,)
        assert y_lower.shape == (100,)

    def test_not_fitted_raises(self, single_output_data):
        X_train, y_train, X_test, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression())
        with pytest.raises(AttributeError):
            cp.predict_interval(X_test)

    def test_all_residuals_computed(self, single_output_data):
        X_train, y_train, _, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        # Every training point should have a residual (no NaNs)
        assert not np.any(np.isnan(cp.residuals_))
        assert cp.residuals_.shape[0] == X_train.shape[0]

    def test_fold_models_stored(self, single_output_data):
        X_train, y_train, _, _ = single_output_data
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
        cp.fit(X_train, y_train)
        assert len(cp.fold_models_) == 5


class TestCVPlusMultiOutput:
    def test_fit_and_predict_interval(self, multi_output_data):
        X_train, y_train, X_test, y_test = multi_output_data
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=0.1)
        assert y_lower.shape == (100, 3)
        assert y_upper.shape == (100, 3)
        assert np.all(y_lower <= y_upper)

    def test_coverage_per_output(self, multi_output_data):
        X_train, y_train, X_test, y_test = multi_output_data
        alpha = 0.1
        cp = CVPlusConformalPredictor(
            RandomForestRegressor(n_estimators=50, random_state=0),
            n_folds=5,
            random_state=0,
        )
        cp.fit(X_train, y_train)
        y_lower, y_upper = cp.predict_interval(X_test, alpha=alpha)
        for j in range(3):
            coverage = np.mean(
                (y_test[:, j] >= y_lower[:, j]) & (y_test[:, j] <= y_upper[:, j])
            )
            assert coverage >= (1 - alpha) - 0.15


def _cv_plus_reference(cp, X, alpha):
    """Direct per-point implementation of the CV+ definition (Barber et al., 2021)."""
    n = cp.residuals_.shape[0]
    preds = [np.asarray(m.predict(X)).reshape(len(X), -1) for m in cp.fold_models_]
    k_lo = int(np.floor(alpha * (n + 1)))
    k_hi = int(np.ceil((1 - alpha) * (n + 1)))
    lo = np.empty((len(X), cp.residuals_.shape[1]))
    hi = np.empty_like(lo)
    for t in range(len(X)):
        for j in range(cp.residuals_.shape[1]):
            lows = sorted(preds[cp.fold_ids_[i]][t, j] - cp.residuals_[i, j] for i in range(n))
            highs = sorted(preds[cp.fold_ids_[i]][t, j] + cp.residuals_[i, j] for i in range(n))
            lo[t, j] = lows[k_lo - 1]
            hi[t, j] = highs[k_hi - 1]
    return lo, hi


def test_matches_reference_definition(multi_output_data):
    # Regression: intervals were mean(fold preds) +/- a residual quantile, which
    # is not CV+ and carries no coverage guarantee.
    X_train, y_train, X_test, _ = multi_output_data
    cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=0)
    cp.fit(X_train, y_train)
    lo, hi = cp.predict_interval(X_test[:20], alpha=0.1)
    ref_lo, ref_hi = _cv_plus_reference(cp, X_test[:20], alpha=0.1)
    np.testing.assert_allclose(lo, ref_lo)
    np.testing.assert_allclose(hi, ref_hi)


def test_too_few_points_gives_infinite_bounds():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(8, 2)), rng.normal(size=8)
    cp = CVPlusConformalPredictor(LinearRegression(), n_folds=4, random_state=0).fit(X, y)
    with pytest.warns(UserWarning, match="too few"):
        lo, hi = cp.predict_interval(X[:3], alpha=0.1)  # floor(0.1 * 9) = 0 < 1
    assert np.all(np.isneginf(lo)) and np.all(np.isposinf(hi))


def test_marginal_coverage_simulation():
    # Guarantee is >= 1 - 2*alpha; in practice CV+ should be close to 1 - alpha.
    rng = np.random.default_rng(1)
    alpha, coverages = 0.1, []
    for rep in range(100):
        X = rng.uniform(-2, 2, size=(250, 1))
        y = X[:, 0] ** 3 + rng.standard_t(3, size=250)
        cp = CVPlusConformalPredictor(LinearRegression(), n_folds=5, random_state=rep)
        cp.fit(X[:50], y[:50])
        lo, hi = cp.predict_interval(X[50:], alpha=alpha)
        coverages.append(np.mean((y[50:] >= lo) & (y[50:] <= hi)))
    assert np.mean(coverages) >= 1 - 2 * alpha
    assert np.mean(coverages) >= 1 - alpha - 0.03
