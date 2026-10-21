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
