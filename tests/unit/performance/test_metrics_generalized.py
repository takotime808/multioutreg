# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from multioutreg.performance.metrics_generalized_api import (
    _predict_with_uncertainty,
    get_uq_performance_metrics_flexible,
)


def test_get_uq_performance_metrics_flexible():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 4)
    Y = np.dot(X, rng.rand(4, 3)) + rng.randn(40, 3) * 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    model = MultiOutputRegressor(GaussianProcessRegressor())
    model.fit(X_train, y_train)

    metrics_df, overall = get_uq_performance_metrics_flexible(model, X_test, y_test)
    assert not metrics_df.empty
    assert {'rmse', 'mae', 'nll', 'miscal_area', 'output'} <= set(metrics_df.columns)
    assert 'mae' in overall

# # --- Mock for uncertainty_toolbox.get_all_metrics ---
# import types

dummy_metrics = {
    'accuracy': {'rmse': 1.2, 'mae': 0.7},
    'avg_calibration': {'miscal_area': 0.1},
    'scoring_rule': {'nll': 2.3}
}

def dummy_get_all_metrics(y_pred, y_std, y_true):
    return dummy_metrics

import multioutreg.performance.metrics_generalized_api as mgapi
mgapi.uct.get_all_metrics = dummy_get_all_metrics

# --- Estimator mocks ---
class EstimatorReturnStd:
    def predict(self, X, return_std=False, return_cov=False):
        if return_std:
            return np.ones((len(X), 1)), np.ones((len(X), 1)) * 0.5
        raise TypeError

class EstimatorReturnCov:
    def predict(self, X, return_std=False, return_cov=False):
        if return_cov:
            n = len(X)
            return np.ones((n, 1)), np.ones((n, 1, 1))
        raise TypeError

class EstimatorPredictStd:
    def predict(self, X, return_std=False, return_cov=False):
        return np.ones((len(X), 1))
    def predict_std(self, X):
        return np.ones((len(X), 1)) * 0.7

class EstimatorPredictVar:
    def predict(self, X, return_std=False, return_cov=False):
        return np.ones((len(X), 1))
    def predict_var(self, X):
        return np.ones((len(X), 1)) * 0.49

class BadEstimator:
    def predict(self, X, return_std=False, return_cov=False):
        raise TypeError

# --- _predict_with_uncertainty tests ---
def test_predict_with_uncertainty_return_std():
    est = EstimatorReturnStd()
    X = np.zeros((4, 2))
    pred, std = _predict_with_uncertainty(est, X, method="return_std")
    assert pred.shape == (4, 1)
    assert std.shape == (4, 1)

def test_predict_with_uncertainty_return_cov():
    est = EstimatorReturnCov()
    X = np.zeros((3, 2))
    pred, std = _predict_with_uncertainty(est, X, method="return_cov")
    assert pred.shape == (3, 1)
    assert std.shape == (3, 1)

def test_predict_with_uncertainty_predict_std():
    est = EstimatorPredictStd()
    X = np.zeros((2, 2))
    pred, std = _predict_with_uncertainty(est, X, method="predict_std")
    assert pred.shape == (2, 1)
    assert std.shape == (2, 1)
    np.testing.assert_allclose(std, 0.7)

def test_predict_with_uncertainty_predict_var():
    est = EstimatorPredictVar()
    X = np.zeros((5, 2))
    pred, std = _predict_with_uncertainty(est, X, method="predict_var")
    assert pred.shape == (5, 1)
    assert std.shape == (5, 1)
    np.testing.assert_allclose(std, 0.7)

def test_predict_with_uncertainty_fallback():
    est = EstimatorPredictStd()
    X = np.zeros((3, 2))
    pred, std = _predict_with_uncertainty(est, X)
    assert pred.shape == (3, 1)
    assert std.shape == (3, 1)

def test_predict_with_uncertainty_error():
    est = BadEstimator()
    X = np.zeros((3, 2))
    with pytest.raises(RuntimeError):
        _predict_with_uncertainty(est, X)

# --- get_uq_performance_metrics_flexible tests ---

class DummyModel:
    def predict(self, X):
        return np.ones((len(X), 2))
    def predict_std(self, X):
        return np.ones((len(X), 2)) * 0.4

def test_get_uq_performance_metrics_manual_std():
    model = DummyModel()
    X = np.zeros((4, 3))
    y = np.ones((4, 2))
    y_pred_std = np.ones((4, 2)) * 0.5
    metrics_df, overall = get_uq_performance_metrics_flexible(model, X, y, y_pred_std=y_pred_std)
    assert isinstance(metrics_df, pd.DataFrame)
    assert isinstance(overall, dict)
    assert 'rmse' in metrics_df.columns
    assert 'mae' in metrics_df.columns

def test_get_uq_performance_metrics_manual_var():
    model = DummyModel()
    X = np.zeros((5, 2))
    y = np.ones((5, 2))
    y_pred_var = np.ones((5, 2)) * 0.16
    metrics_df, overall = get_uq_performance_metrics_flexible(model, X, y, y_pred_std=y_pred_var, std_is_var=True)
    assert np.allclose(metrics_df['rmse'], 1.2)
    assert 'miscal_area' in metrics_df.columns

def test_get_uq_performance_metrics_predict_std():
    model = DummyModel()
    X = np.zeros((6, 2))
    y = np.ones((6, 2))
    metrics_df, overall = get_uq_performance_metrics_flexible(model, X, y, uncertainty_method="predict_std")
    assert 'nll' in overall

def test_get_uq_performance_metrics_shape_mismatch():
    model = DummyModel()
    X = np.zeros((2, 2))
    y = np.ones((2, 3)) # wrong shape
    with pytest.raises(ValueError):
        get_uq_performance_metrics_flexible(model, X, y, uncertainty_method="predict_std")

def test_get_uq_performance_metrics_runtime_error():
    class BadModel:
        def predict(self, X):
            raise Exception("nope")
    model = BadModel()
    X = np.zeros((2, 2))
    y = np.ones((2, 2))
    with pytest.raises(RuntimeError):
        get_uq_performance_metrics_flexible(model, X, y)

class EstimatorForMulti:
    def predict(self, X, return_std=False, return_cov=False):
        if return_std:
            return np.ones((len(X),)), np.ones((len(X),))
        return np.ones((len(X),))
    def predict_std(self, X):
        return np.ones((len(X),))

def test_get_uq_performance_metrics_multioutput():
    base = EstimatorForMulti()
    model = MultiOutputRegressor(base)
    # Fit model to populate .estimators_ (simulate after fit)
    model.estimators_ = [EstimatorForMulti(), EstimatorForMulti()]
    X = np.zeros((4, 2))
    y = np.ones((4, 2))
    metrics_df, overall = get_uq_performance_metrics_flexible(model, X, y, uncertainty_method="return_std")
    assert 'mae' in overall and 'rmse' in metrics_df.columns
