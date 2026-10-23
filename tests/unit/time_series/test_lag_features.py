# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.lag_features import make_lag_features, rolling_window_features


class TestMakeLagFeatures:

    def test_single_step_shapes(self):
        y = np.arange(50, dtype=float)
        X, tgt = make_lag_features(y, n_lags=5, horizon=1)
        # n_samples = 50 - 5 - 1 + 1 = 45
        assert X.shape == (45, 5)
        assert tgt.shape == (45, 1)

    def test_multi_step_shapes(self):
        y = np.arange(60, dtype=float)
        X, tgt = make_lag_features(y, n_lags=5, horizon=3)
        # n_samples = 60 - 5 - 3 + 1 = 53
        assert X.shape == (53, 5)
        assert tgt.shape == (53, 3)

    def test_lag_values_correct(self):
        """First row of X should be [0,1,2,3,4]; first target y[0,0] = 5."""
        y = np.arange(20, dtype=float)
        X, tgt = make_lag_features(y, n_lags=5, horizon=1)
        np.testing.assert_array_equal(X[0], [0.0, 1.0, 2.0, 3.0, 4.0])
        assert tgt[0, 0] == 5.0

    def test_multi_step_target_values_correct(self):
        """With horizon=3, first target row should be [5, 6, 7]."""
        y = np.arange(20, dtype=float)
        X, tgt = make_lag_features(y, n_lags=5, horizon=3)
        np.testing.assert_array_equal(tgt[0], [5.0, 6.0, 7.0])

    def test_accepts_pd_series(self):
        y = pd.Series(np.arange(30, dtype=float))
        X, tgt = make_lag_features(y, n_lags=4, horizon=2)
        assert X.shape[1] == 4
        assert tgt.shape[1] == 2

    def test_time_features_expand_columns(self):
        """include_time_features=True with DatetimeIndex adds 4 extra columns."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        y = pd.Series(np.arange(50, dtype=float), index=dates)
        X_no, _ = make_lag_features(y, n_lags=5, horizon=1, include_time_features=False)
        X_yes, _ = make_lag_features(y, n_lags=5, horizon=1, include_time_features=True)
        assert X_yes.shape[1] == X_no.shape[1] + 4

    def test_too_short_raises(self):
        y = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="too short"):
            make_lag_features(y, n_lags=10, horizon=1)

    def test_no_nans_in_output(self):
        y = np.sin(np.linspace(0, 4 * np.pi, 100))
        X, tgt = make_lag_features(y, n_lags=10, horizon=5)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(tgt))


class TestRollingWindowFeatures:

    def test_shape_two_windows(self):
        """2 windows → 4 columns (mean + std each)."""
        y = np.arange(50, dtype=float)
        feats = rolling_window_features(y, windows=(4, 8))
        assert feats.shape == (50, 4)

    def test_shape_default_windows(self):
        """Default (4, 8, 24) → 6 columns."""
        y = np.arange(50, dtype=float)
        feats = rolling_window_features(y)
        assert feats.shape == (50, 6)

    def test_no_leading_nans(self):
        """min_periods=1 ensures no NaN at the start."""
        y = np.arange(20, dtype=float)
        feats = rolling_window_features(y, windows=(4,))
        assert not np.any(np.isnan(feats))

    def test_first_mean_is_first_value(self):
        """With window=4 and min_periods=1, rolling mean of [0,1,2,...] at index 0 = 0."""
        y = np.arange(20, dtype=float)
        feats = rolling_window_features(y, windows=(4,))
        assert feats[0, 0] == pytest.approx(0.0)
