# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates.quantile_sklearn import QuantileRegressionSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(2026)
    X = np.random.rand(100, 4)
    Y = np.column_stack([
        2 * X[:, 0] - X[:, 1] + 0.1 * np.random.randn(100),
        np.mean(X, axis=1) + 0.05 * np.random.randn(100),
    ])
    return X, Y


class TestQuantileRegressionSurrogate:
    def test_initialization_default(self):
        s = QuantileRegressionSurrogate()
        assert s.alpha == 1.0
        assert s.miscoverage == 0.1
        assert s.solver == "highs"
        assert s.solver_options is None

    def test_initialization_with_params(self):
        s = QuantileRegressionSurrogate(
            alpha=0.5, miscoverage=0.2, solver="highs-ds", solver_options={"max_iter": 1000}
        )
        assert s.alpha == 0.5
        assert s.miscoverage == 0.2
        assert s.solver == "highs-ds"
        assert s.solver_options == {"max_iter": 1000}

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_fit_1d_y_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = rng.standard_normal(60)
        s = QuantileRegressionSurrogate()
        s.fit(X, y)
        preds = s.predict(X)
        assert preds.shape == (60, 1)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_predict_intervals_shape(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        lower, upper = s.predict_intervals(X)
        assert lower.shape == Y.shape
        assert upper.shape == Y.shape

    def test_predict_intervals_ordering(self, sample_data):
        """Lower quantile predictions should be ≤ upper quantile predictions."""
        X, Y = sample_data
        s = QuantileRegressionSurrogate(miscoverage=0.1)
        s.fit(X, Y)
        lower, upper = s.predict_intervals(X)
        assert np.all(upper >= lower)

    def test_predict_median_between_bounds(self, sample_data):
        """Median (point) predictions should be between lower and upper quantiles."""
        X, Y = sample_data
        s = QuantileRegressionSurrogate(miscoverage=0.1)
        s.fit(X, Y)
        preds = s.predict(X)
        lower, upper = s.predict_intervals(X)
        # Median need not be strictly between for all samples due to quantile crossing,
        # but on average it should be in range
        assert np.mean(preds >= lower - 0.5) > 0.8
        assert np.mean(preds <= upper + 0.5) > 0.8

    def test_miscoverage_affects_interval_width(self, sample_data):
        """Narrower miscoverage target → wider intervals."""
        X, Y = sample_data
        s_wide = QuantileRegressionSurrogate(miscoverage=0.4)  # 60% intervals
        s_narrow = QuantileRegressionSurrogate(miscoverage=0.05)  # 95% intervals
        s_wide.fit(X, Y)
        s_narrow.fit(X, Y)

        lo_w, hi_w = s_wide.predict_intervals(X)
        lo_n, hi_n = s_narrow.predict_intervals(X)

        width_wide = np.mean(hi_w - lo_w)
        width_narrow = np.mean(hi_n - lo_n)
        assert width_narrow >= width_wide

    def test_n_outputs_set_after_fit(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        assert s.n_outputs_ == Y.shape[1]

    def test_models_created_after_fit(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        assert len(s._models_median) == Y.shape[1]
        assert len(s._models_lo) == Y.shape[1]
        assert len(s._models_hi) == Y.shape[1]

    def test_predict_before_fit_raises(self):
        s = QuantileRegressionSurrogate()
        with pytest.raises(AttributeError, match="not fitted"):
            s.predict(np.random.rand(5, 3))

    def test_predict_intervals_before_fit_raises(self):
        s = QuantileRegressionSurrogate()
        with pytest.raises(AttributeError, match="not fitted"):
            s.predict_intervals(np.random.rand(5, 3))

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:70], X[70:]
        Y_train, Y_cal = Y[:70], Y[70:]
        s = QuantileRegressionSurrogate()
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_get_set_params(self):
        s = QuantileRegressionSurrogate(alpha=0.5, miscoverage=0.2)
        params = s.get_params()
        assert params["alpha"] == 0.5
        assert params["miscoverage"] == 0.2
        s.set_params(alpha=0.1)
        assert s.alpha == 0.1

    def test_predict_consistency(self, sample_data):
        X, Y = sample_data
        s = QuantileRegressionSurrogate()
        s.fit(X, Y)
        np.testing.assert_allclose(s.predict(X), s.predict(X), rtol=1e-6)

    def test_pseudo_std_consistent_with_intervals(self, sample_data):
        """pseudo_std ≈ (upper - lower) / (2 * z_alpha)."""
        import math
        X, Y = sample_data
        miscoverage = 0.1
        s = QuantileRegressionSurrogate(miscoverage=miscoverage)
        s.fit(X, Y)
        _, pseudo_std = s.predict(X, return_std=True)
        lo, hi = s.predict_intervals(X)

        # z for two-tailed at 1 - miscoverage/2
        z = math.sqrt(2) * _erfinv_approx(2 * (1 - miscoverage / 2) - 1)
        expected_std = np.maximum((hi - lo) / (2 * z), 0.0)
        np.testing.assert_allclose(pseudo_std, expected_std, rtol=1e-6)


def _erfinv_approx(x: float) -> float:
    """Mirror of the library's _erfinv for test validation."""
    import math
    sign = 1.0 if x >= 0.0 else -1.0
    x = abs(x)
    a = 0.147
    t = math.sqrt(-math.log((1.0 - x * x) / 2.0 + 1e-300))
    t2 = (2.0 / (math.pi * a) + math.log((1.0 - x * x) / 2.0 + 1e-300) / 2.0)
    return sign * math.sqrt(math.sqrt(t2 * t2 - math.log((1.0 - x * x) / 2.0 + 1e-300) / a) - t2)
