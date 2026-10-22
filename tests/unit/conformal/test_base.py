# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from multioutreg.conformal import SplitConformalPredictor
from multioutreg.conformal.base import BaseConformalPredictor


def test_conformal_quantile_is_exact_order_statistic():
    # k = ceil((n+1)(1-alpha)) = ceil(20 * 0.9) = 18 -> 18th smallest score
    scores = np.random.default_rng(0).permutation(np.arange(1.0, 20.0))
    assert BaseConformalPredictor._conformal_quantile(scores, 0.1) == 18.0


@pytest.mark.parametrize("n, alpha", [(8, 0.1), (10, 0.05), (98, 0.01)])
def test_conformal_quantile_too_few_points_is_infinite(n, alpha):
    # Regression: previously clamped to the max score, which under-covers.
    scores = np.arange(1.0, n + 1)
    with pytest.warns(UserWarning, match="too few"):
        assert BaseConformalPredictor._conformal_quantile(scores, alpha) == np.inf


@pytest.mark.parametrize("n, alpha", [(9, 0.1), (19, 0.05), (99, 0.01)])
def test_conformal_quantile_minimum_points_is_finite(n, alpha):
    scores = np.arange(1.0, n + 1)
    assert BaseConformalPredictor._conformal_quantile(scores, alpha) == float(n)


def test_split_conformal_marginal_coverage_small_calibration_set():
    # With n_cal = 8 < 9 the interval must be unbounded; the old clamp gave
    # ~0.886 average coverage for a 0.90 target. Coverage must be >= 1 - alpha.
    rng = np.random.default_rng(0)
    alpha, coverages = 0.1, []
    for rep in range(200):
        X = rng.uniform(-2, 2, size=(540, 1))
        y = X[:, 0] ** 3 + rng.standard_t(3, size=540)
        cp = SplitConformalPredictor(LinearRegression(), calibration_size=0.2, random_state=rep)
        with pytest.warns(UserWarning, match="too few"):
            cp.fit(X[:40], y[:40])  # 32 train / 8 calibration
            lo, hi = cp.predict_interval(X[40:], alpha=alpha)
        coverages.append(np.mean((y[40:] >= lo) & (y[40:] <= hi)))
    assert np.mean(coverages) >= 1 - alpha
