"""Finite-sample checks independent of any fitted forecasting model."""
import numpy as np
import pytest
from multioutreg.time_series.uncertainty import conformal_interval_from_residuals


def test_uses_order_statistic_without_interpolation():
    lower, upper = conformal_interval_from_residuals([10.], [0., 1., 2., 100.], alpha=.4)
    np.testing.assert_array_equal(lower, [8.])
    np.testing.assert_array_equal(upper, [12.])


def test_insufficient_calibration_returns_unbounded_interval():
    lower, upper = conformal_interval_from_residuals([0.], [1., 2.], alpha=.1)
    assert np.isneginf(lower).all() and np.isposinf(upper).all()


@pytest.mark.parametrize('n,alpha', [(4, .4), (9, .1), (2, .1), (19, .25)])
def test_exact_exchangeable_rank_coverage(n, alpha):
    # Enumerate which of n+1 distinct scores is the held-out observation.
    scores = np.arange(n + 1, dtype=float)
    covered = 0
    for i, score in enumerate(scores):
        _, upper = conformal_interval_from_residuals([0.], np.delete(scores, i), alpha)
        covered += score <= upper[0]
    assert covered / (n + 1) >= 1 - alpha


@pytest.mark.parametrize('residuals', [[], [[1., 2.]], [np.nan], [np.inf]])
def test_invalid_calibration_is_rejected(residuals):
    with pytest.raises(ValueError, match='cal_residuals'):
        conformal_interval_from_residuals([0.], residuals)


@pytest.mark.parametrize('alpha', [0., 1., -.1, np.nan, np.inf])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match='alpha'):
        conformal_interval_from_residuals([0.], [1.], alpha)
