# Copyright (c) 2026 takotime808

"""Audit item 29: get_uq_performance_metrics_flexible API selection and pooled metrics.

Kept apart from test_metrics_generalized.py, which mocks uncertainty-toolbox.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.performance.metrics_generalized_api import (
    _predict_with_uncertainty,
    get_uq_performance_metrics_flexible,
)


class _ConstStd:
    def __init__(self, std):
        self.std = std

    def predict(self, X, return_std=False):
        pred = np.zeros(len(X))
        return (pred, np.full(len(X), self.std)) if return_std else pred


class _MORWithOwnStd(MultiOutputRegressor):
    """A MultiOutputRegressor whose own predict() also accepts return_std."""

    def predict(self, X, return_std=False):
        pred = np.zeros((len(X), 2))
        return (pred, np.full((len(X), 2), 99.0)) if return_std else pred


def test_per_estimator_std_not_overwritten_by_return_std_branch():
    # The return_std branch used to run even after the per-estimator pass succeeded.
    model = _MORWithOwnStd(_ConstStd(1.0))
    model.estimators_ = [_ConstStd(1.0), _ConstStd(2.0)]
    X = np.zeros((20, 1))
    y = np.random.default_rng(0).standard_normal((20, 2))
    metrics_df, _ = get_uq_performance_metrics_flexible(
        model, X, y, uncertainty_method="return_std"
    )
    ref, _ = get_uq_performance_metrics_flexible(
        model, X, y, y_pred_std=np.tile([1.0, 2.0], (20, 1))
    )
    np.testing.assert_allclose(metrics_df["nll"], ref["nll"])


def test_predict_var_called_once():
    class VarOnly:
        calls = 0

        def predict(self, X):
            return np.zeros((len(X), 1))

        def predict_var(self, X):
            VarOnly.calls += 1
            return np.full((len(X), 1), 4.0)

    X = np.zeros((10, 1))
    y = np.random.default_rng(1).standard_normal((10, 1))
    get_uq_performance_metrics_flexible(VarOnly(), X, y, uncertainty_method="predict_var")
    assert VarOnly.calls == 1


def test_overall_metrics_do_not_depend_on_output_units():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(80, 2))
    Y = np.column_stack([X @ [1.0, 2.0], X @ [-1.0, 0.5]]) + 0.1 * rng.standard_normal((80, 2))
    pred = Y + 0.1 * rng.standard_normal(Y.shape)
    std = np.full(Y.shape, 0.1)

    class Fixed:
        def __init__(self, p):
            self.p = p

        def predict(self, X):
            return self.p

    _, base = get_uq_performance_metrics_flexible(Fixed(pred), X, Y, y_pred_std=std)
    units = np.array([1.0, 1000.0])  # second output in different units
    _, scaled = get_uq_performance_metrics_flexible(
        Fixed(pred * units), X, Y * units, y_pred_std=std * units
    )
    for key in ("rmse", "mae", "nll", "miscal_area"):
        np.testing.assert_allclose(scaled[key], base[key], rtol=1e-6)


def test_return_cov_from_sklearn_gp_multi_target():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(15, 1))
    Y = np.column_stack([np.sin(3 * X[:, 0]), np.cos(3 * X[:, 0])])
    gp = GaussianProcessRegressor(alpha=1e-2).fit(X, Y)
    _, std_ref = gp.predict(X[:5], return_std=True)
    pred, std = _predict_with_uncertainty(gp, X[:5], method="return_cov")
    assert std.shape == pred.shape == (5, 2)
    np.testing.assert_allclose(std, std_ref, rtol=1e-6, atol=1e-8)


def test_plain_gp_without_method_now_works():
    # Previously only predict_std / predict_var were tried when no method was given.
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(30, 2))
    Y = np.column_stack([X.sum(1), X[:, 0] - X[:, 1]])
    gp = GaussianProcessRegressor(alpha=1e-2).fit(X[:20], Y[:20])
    metrics_df, overall = get_uq_performance_metrics_flexible(gp, X[20:], Y[20:])
    assert len(metrics_df) == 2 and np.isfinite(overall["nll"])
