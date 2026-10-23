# Copyright (c) 2025 takotime808

import numpy as np
import pytest

from multioutreg.time_series.uncertainty import (
    gaussian_quantiles,
    conformal_interval_from_residuals,
    propagate_uncertainty_recursive,
)


class TestGaussianQuantiles:

    def test_shape(self):
        mean = np.zeros(10)
        std = np.ones(10)
        q = gaussian_quantiles(mean, std, quantiles=(0.1, 0.5, 0.9))
        assert q.shape == (3, 10)

    def test_median_equals_mean(self):
        """Median quantile (q=0.5) should equal the mean."""
        mean = np.arange(5, dtype=float)
        std = np.ones(5)
        q = gaussian_quantiles(mean, std, quantiles=(0.5,))
        np.testing.assert_allclose(q[0], mean, atol=1e-9)

    def test_ordering(self):
        """Lower quantiles should be ≤ upper quantiles at every horizon step."""
        rng = np.random.default_rng(0)
        mean = rng.standard_normal(8)
        std = np.abs(rng.standard_normal(8)) + 0.1
        q = gaussian_quantiles(mean, std, quantiles=(0.1, 0.5, 0.9))
        assert np.all(q[0] <= q[1] + 1e-12)
        assert np.all(q[1] <= q[2] + 1e-12)

    def test_symmetry(self):
        """q=0.1 and q=0.9 should be symmetric around q=0.5 for Gaussian."""
        mean = np.array([0.0, 1.0, 2.0])
        std = np.array([1.0, 1.0, 1.0])
        q = gaussian_quantiles(mean, std, quantiles=(0.1, 0.5, 0.9))
        np.testing.assert_allclose(q[0], 2 * q[1] - q[2], atol=1e-9)

    def test_zero_std_all_same(self):
        """With std=0 all quantiles equal the mean."""
        mean = np.array([3.0, 4.0])
        std = np.zeros(2)
        q = gaussian_quantiles(mean, std, quantiles=(0.1, 0.5, 0.9))
        np.testing.assert_allclose(q[0], mean)
        np.testing.assert_allclose(q[2], mean)


class TestConformalIntervalFromResiduals:

    def test_interval_contains_calibration_fraction(self):
        """The interval should contain at least (1-alpha) of calibration residuals."""
        rng = np.random.default_rng(42)
        cal_residuals = rng.standard_normal(200)
        alpha = 0.1
        point = np.zeros(5)
        lower, upper = conformal_interval_from_residuals(point, cal_residuals, alpha=alpha)
        covered = np.mean((cal_residuals >= lower[0]) & (cal_residuals <= upper[0]))
        assert covered >= (1 - alpha) - 0.05

    def test_lower_leq_upper(self):
        cal_residuals = np.abs(np.random.default_rng(0).standard_normal(100))
        point = np.arange(5, dtype=float)
        lower, upper = conformal_interval_from_residuals(point, cal_residuals)
        assert np.all(lower <= upper + 1e-9)

    def test_shape(self):
        cal_residuals = np.arange(50, dtype=float)
        point = np.zeros(7)
        lower, upper = conformal_interval_from_residuals(point, cal_residuals)
        assert lower.shape == (7,)
        assert upper.shape == (7,)

    def test_wider_with_larger_alpha(self):
        """Smaller alpha (more coverage) → wider interval."""
        rng = np.random.default_rng(0)
        cal_residuals = rng.standard_normal(200)
        point = np.zeros(3)
        lower_05, upper_05 = conformal_interval_from_residuals(point, cal_residuals, alpha=0.05)
        lower_20, upper_20 = conformal_interval_from_residuals(point, cal_residuals, alpha=0.20)
        width_05 = (upper_05 - lower_05).mean()
        width_20 = (upper_20 - lower_20).mean()
        assert width_05 >= width_20


class TestPropagateUncertaintyRecursive:

    def test_shape(self):
        stds = propagate_uncertainty_recursive(1.0, horizon=10)
        assert stds.shape == (10,)

    def test_white_noise_grows_as_sqrt_h(self):
        """With correlation=0, sigma_h = sigma_1 * sqrt(h)."""
        sigma1 = 2.0
        stds = propagate_uncertainty_recursive(sigma1, horizon=5, correlation=0.0)
        expected = sigma1 * np.sqrt(np.arange(1, 6))
        np.testing.assert_allclose(stds, expected, rtol=1e-9)

    def test_positive_correlation_uncertainty_grows(self):
        """AR(1) uncertainty is monotonically non-decreasing with horizon."""
        sigma1 = 1.0
        stds = propagate_uncertainty_recursive(sigma1, horizon=5, correlation=0.8)
        # Each step's uncertainty should be >= the previous step's
        assert np.all(np.diff(stds) >= -1e-9)

    def test_first_step_equals_sigma1(self):
        """Horizon 1 std should always equal sigma_1."""
        sigma1 = 3.5
        stds = propagate_uncertainty_recursive(sigma1, horizon=4, correlation=0.5)
        assert stds[0] == pytest.approx(sigma1, rel=1e-6)

    def test_non_negative(self):
        stds = propagate_uncertainty_recursive(1.0, horizon=8, correlation=-0.5)
        assert np.all(stds >= 0)
