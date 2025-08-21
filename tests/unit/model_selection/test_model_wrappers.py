# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from multioutreg.model_selection.model_wrappers import PerTargetRegressorWithStd


@pytest.fixture
def sample_data():
    X = np.random.rand(20, 3)
    y = np.random.rand(20, 2)
    return X, y


def test_fit_and_predict_without_std(sample_data):
    X, y = sample_data
    estimators = [LinearRegression(), LinearRegression()]
    model = PerTargetRegressorWithStd(estimators)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) == y.shape[1]


def test_fit_and_predict_with_std_supported(sample_data):
    X, y = sample_data
    estimators = [GaussianProcessRegressor(), GaussianProcessRegressor()]
    model = PerTargetRegressorWithStd(estimators)
    model.fit(X, y)

    y_pred, y_std = model.predict(X, return_std=True)
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_std, np.ndarray)
    assert y_pred.shape == y_std.shape == y.shape
    assert not np.isnan(y_std).any()


def test_fit_and_predict_with_std_unsupported(sample_data):
    X, y = sample_data
    estimators = [LinearRegression(), LinearRegression()]
    model = PerTargetRegressorWithStd(estimators)
    model.fit(X, y)

    y_pred, y_std = model.predict(X, return_std=True)
    assert y_pred.shape == y_std.shape == y.shape
    assert np.isnan(y_std).all()


def test_fit_with_1d_target():
    X = np.random.rand(20, 3)
    y = np.random.rand(20)  # 1D target
    estimators = [LinearRegression()]
    model = PerTargetRegressorWithStd(estimators)
    model.fit(X, y)

    y_pred = model.predict(X)
    assert y_pred.shape == (20, 1)


def test_predict_without_fit_raises():
    X = np.random.rand(10, 3)
    estimators = [LinearRegression(), LinearRegression()]
    model = PerTargetRegressorWithStd(estimators)
    with pytest.raises(AttributeError):
        model.predict(X)
