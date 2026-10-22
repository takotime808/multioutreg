# Copyright (c) 2026 takotime808

import numpy as np

from multioutreg.model_selection.screening import _blr


def test_bootstrap_lr_keeps_x_y_pairs():
    # Regression: X and y were resampled with independent indices, which
    # destroyed the X-y relationship and gave slopes near zero.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = 3.0 * X[:, 0] + 0.1 * rng.normal(size=200)
    model = _blr().fit(X, y)
    slopes = np.array([m.coef_[0] for m in model.models_])
    np.testing.assert_allclose(slopes, 3.0, atol=0.1)


def test_gbq_std_is_sigma_not_half_width():
    # Regression: std was the half-width of the 90% quantile interval
    # (~1.645 * sigma for Gaussian noise) instead of sigma itself.
    from multioutreg.model_selection.screening import _gbq

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 1))
    y = X[:, 0] + rng.normal(size=2000)  # sigma = 1
    _, std = _gbq().fit(X, y).predict(X[:500], return_std=True)
    assert abs(std.mean() - 1.0) < 0.2
