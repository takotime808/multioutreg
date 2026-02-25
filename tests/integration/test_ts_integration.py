# Copyright (c) 2025 takotime808
"""Integration test: walk-forward evaluation of LagFeatureForecaster on an AR(1) series.

This test verifies end-to-end that:
  - LagFeatureForecaster with RandomForest produces n_folds >= 10
  - The 80% prediction interval covers >= 70% of held-out values

No external dependencies beyond sklearn are required.
"""

import numpy as np
import pytest

from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------------------
# AR(1) series generator
# ---------------------------------------------------------------------------

def _ar1_series(n: int = 300, rho: float = 0.7, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate a stationary AR(1) series y_t = rho * y_{t-1} + eps_t."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    y[0] = rng.normal(0, sigma / np.sqrt(1 - rho ** 2))
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal(0, sigma)
    return y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLagFeatureForecasterIntegration:

    def test_walk_forward_n_folds(self):
        """At least 10 folds should be generated for a 300-step series."""
        from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
        from multioutreg.time_series.cv import WalkForwardCV

        y = _ar1_series(n=300)
        lff = LagFeatureForecaster(
            surrogate=RandomForestRegressor(n_estimators=50, random_state=0),
            n_lags=12,
            horizon=5,
            uncertainty="return_std",
        )
        cv = WalkForwardCV(min_train=60, horizon=5, step=10)
        summary = cv.summary(cv.evaluate(y, lff))

        assert summary["n_folds"] >= 10, f"Expected >=10 folds, got {summary['n_folds']}"

    def test_walk_forward_interval_coverage(self):
        """80% prediction interval should cover at least 70% of held-out values.

        Uses uncertainty='conformal' with a residual-based fallback so that any
        sklearn estimator (not just GP models with return_std) produces intervals.
        """
        from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
        from multioutreg.time_series.cv import WalkForwardCV

        y = _ar1_series(n=300)
        lff = LagFeatureForecaster(
            surrogate=RandomForestRegressor(n_estimators=50, random_state=0),
            n_lags=12,
            horizon=5,
            uncertainty="conformal",
        )
        cv = WalkForwardCV(min_train=60, horizon=5, step=10)
        folds = cv.evaluate(y, lff)

        covered = 0
        total = 0
        for fold in folds:
            if fold.quantiles is None:
                continue
            lower = fold.quantiles[0, :]   # q=0.1
            upper = fold.quantiles[2, :]   # q=0.9
            y_true = fold.y_true
            covered += int(np.sum((y_true >= lower) & (y_true <= upper)))
            total += len(y_true)

        if total == 0:
            pytest.skip("No quantile predictions available (uncertainty='none' or no folds)")

        coverage = covered / total
        assert coverage >= 0.70, (
            f"80% interval coverage too low: {coverage:.2%} < 70% "
            f"(covered={covered}, total={total})"
        )

    def test_final_forecast_shape(self):
        """Final fit and predict should return ForecastResult with correct shape."""
        from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
        from multioutreg.time_series.chronos_adapter import ForecastResult

        y = _ar1_series(n=200)
        lff = LagFeatureForecaster(
            surrogate=RandomForestRegressor(n_estimators=30, random_state=0),
            n_lags=10,
            horizon=8,
            uncertainty="return_std",
        )
        lff.fit(y)
        res = lff.predict(horizon=8, quantiles=(0.1, 0.5, 0.9))

        assert isinstance(res, ForecastResult)
        assert res.quantiles.shape == (1, 3, 8)

    def test_summary_keys_present(self):
        """WalkForwardCV.summary() should contain expected metric keys."""
        from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
        from multioutreg.time_series.cv import WalkForwardCV

        y = _ar1_series(n=150)
        lff = LagFeatureForecaster(
            surrogate=RandomForestRegressor(n_estimators=30, random_state=0),
            n_lags=8,
            horizon=3,
            uncertainty="none",
        )
        cv = WalkForwardCV(min_train=40, horizon=3, step=5)
        summary = cv.summary(cv.evaluate(y, lff))

        for key in ("n_folds", "mean_smape", "std_smape", "mean_mase", "std_mase"):
            assert key in summary, f"Missing key: {key}"

    def test_conformal_uncertainty_produces_intervals(self):
        """uncertainty='conformal' should yield non-degenerate prediction intervals."""
        from multioutreg.time_series.lag_forecaster import LagFeatureForecaster

        y = _ar1_series(n=150)
        lff = LagFeatureForecaster(
            surrogate=RandomForestRegressor(n_estimators=30, random_state=0),
            n_lags=8,
            horizon=3,
            uncertainty="conformal",
        )
        lff.fit(y)
        res = lff.predict(horizon=3, quantiles=(0.1, 0.5, 0.9))

        q = res.quantiles[0]  # (3, 3)
        # lower ≤ upper everywhere
        assert np.all(q[0] <= q[2] + 1e-9)
        # At least the interval should be non-trivially zero-width at some step
        assert np.any(q[2] - q[0] > 0), "All prediction intervals have zero width"
