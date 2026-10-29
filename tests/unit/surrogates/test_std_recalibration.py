# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.base import clone

from multioutreg.surrogates import BayesianRidgeSurrogate, StdRecalibratedSurrogate


class _ScaledStd(BayesianRidgeSurrogate):
    """BayesianRidge with its std multiplied by a fixed factor, to miscalibrate it on purpose."""

    def __init__(self, factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def predict(self, X, return_std=False):
        if not return_std:
            return super().predict(X)
        mean, std = super().predict(X, return_std=True)
        return mean, std * self.factor


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - 2 * X[:, 1]]) + rng.normal(scale=0.3, size=(n, 2))
    return X, Y


@pytest.mark.parametrize("criterion", ["nll", "ma_cal"])
def test_undoes_a_known_miscalibration(criterion):
    X, Y = _data()
    model = StdRecalibratedSurrogate(_ScaledStd, criterion=criterion, random_state=0, factor=0.25).fit(X, Y)
    # The wrapped std is 4x too small, so the learned factors should be near 4.
    np.testing.assert_allclose(model.std_scale_, 4.0, rtol=0.25)


def test_recalibrated_std_reaches_nominal_coverage():
    X, Y = _data(n=600)
    X_tr, Y_tr, X_te, Y_te = X[:300], Y[:300], X[300:], Y[300:]
    model = StdRecalibratedSurrogate(_ScaledStd, criterion="nll", random_state=0, factor=3.0).fit(X_tr, Y_tr)
    mean, std = model.predict(X_te, return_std=True)
    coverage = np.mean(np.abs(Y_te - mean) <= 1.96 * std)
    assert coverage == pytest.approx(0.95, abs=0.04)


def test_mean_unchanged_and_recalibrate_on_holdout():
    X, Y = _data()
    model = StdRecalibratedSurrogate(_ScaledStd, criterion="nll", random_state=0, factor=0.5).fit(X[:200], Y[:200])
    base = _ScaledStd(factor=0.5).fit(X[:200], Y[:200])
    np.testing.assert_allclose(model.predict(X[200:]), base.predict(X[200:]))
    before = model.std_scale_.copy()
    model.recalibrate(X[200:], Y[200:])
    assert model.std_scale_.shape == (2,)
    np.testing.assert_allclose(model.std_scale_, before, rtol=0.3)


def test_rejects_bad_arguments_and_clones():
    with pytest.raises(ValueError):
        StdRecalibratedSurrogate(criterion="bogus")
    X, Y = _data(n=10)
    with pytest.raises(ValueError):
        StdRecalibratedSurrogate(BayesianRidgeSurrogate, cv=1).fit(X, Y)
    model = StdRecalibratedSurrogate(BayesianRidgeSurrogate, criterion="nll", cv=3)
    assert clone(model).get_params() == model.get_params()
