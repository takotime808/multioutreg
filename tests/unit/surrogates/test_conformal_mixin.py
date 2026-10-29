# Copyright (c) 2026 takotime808

"""ConformalMixin on a minimal surrogate."""

import numpy as np
import pytest

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class _Shift(ConformalMixin):
    """Predicts ``x + shift``; 1-D output, to exercise the reshape."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def predict(self, X):
        return np.asarray(X)[:, 0] + self.shift


def test_conformal_predict_before_wrap_raises():
    with pytest.raises(AttributeError, match="wrap_conformal"):
        _Shift().conformal_predict(np.zeros((2, 1)))


def test_1d_predictions_and_targets_are_treated_as_one_output():
    rng = np.random.default_rng(0)
    X_cal = rng.uniform(size=(200, 1))
    y_cal = X_cal[:, 0] + rng.uniform(-1, 1, size=200)  # residuals ~ U(0, 1)
    model = _Shift().wrap_conformal(X_cal, y_cal)
    assert model._conformal_residuals.shape == (200, 1)

    X = np.array([[0.0], [2.0]])
    lower, upper = model.conformal_predict(X, alpha=0.2)
    assert lower.shape == upper.shape == (2, 1)
    half = (upper - lower) / 2
    np.testing.assert_allclose(half[0], half[1])
    np.testing.assert_allclose((upper + lower)[:, 0] / 2, X[:, 0])
    assert 0.7 < half[0, 0] < 0.9  # ~ the 80% quantile of U(0, 1)


def test_intervals_cover_fresh_data():
    rng = np.random.default_rng(1)
    model = _Shift(shift=0.5)
    X_cal, X_new = rng.normal(size=(500, 1)), rng.normal(size=(2000, 1))
    noise = lambda n: 0.5 + rng.normal(scale=0.3, size=n)
    model.wrap_conformal(X_cal, (X_cal[:, 0] + noise(500)).reshape(-1, 1))
    lower, upper = model.conformal_predict(X_new, alpha=0.1)
    y_new = X_new[:, 0] + noise(2000)
    coverage = np.mean((lower[:, 0] <= y_new) & (y_new <= upper[:, 0]))
    assert 0.87 < coverage < 0.93
