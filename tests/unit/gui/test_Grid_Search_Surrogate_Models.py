# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
from multioutreg.gui.Grid_Search_Surrogate_Models import (
    RandomForestWithUncertainty,
    GradientBoostingWithUncertainty,
    KNeighborsRegressorWithUncertainty,
    BootstrapLinearRegression,
    PerTargetRegressorWithStd
)
from multioutreg.surrogates import MultiFidelitySurrogate, LinearRegressionSurrogate

@pytest.fixture
def sample_data():
    X, y = make_regression(n_samples=100, n_features=5, n_targets=3, noise=0.1, random_state=42)
    return X, y

def test_random_forest_with_uncertainty(sample_data):
    X, y = sample_data
    model = RandomForestWithUncertainty(n_estimators=10)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert np.all(std >= 0)

def test_gradient_boosting_with_uncertainty(sample_data):
    X, y = sample_data
    model = GradientBoostingWithUncertainty(n_estimators=10)
    model.fit(X, y[:, 0])  # single output
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)
    assert np.all(std >= 0)

def test_knn_with_uncertainty(sample_data):
    X, y = sample_data
    model = KNeighborsRegressorWithUncertainty(n_neighbors=3)
    model.fit(X, y[:, 0])
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)
    assert np.all(std >= 0)

def test_bootstrap_linear_regression(sample_data):
    X, y = sample_data
    model = BootstrapLinearRegression(n_bootstraps=5)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert np.all(std >= 0)

def test_per_target_regressor_with_std(sample_data):
    X, y = sample_data
    base_models = [RandomForestWithUncertainty(n_estimators=5) for _ in range(y.shape[1])]
    model = PerTargetRegressorWithStd(base_models)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape
    assert np.all(std >= 0)

def test_multi_fidelity_surrogate_single_level(sample_data):
    X, y = sample_data
    mfs = MultiFidelitySurrogate(LinearRegressionSurrogate, ["default"])
    mfs.fit((X, y))
    preds = mfs.predict(X)
    assert preds.shape == y.shape