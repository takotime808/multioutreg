# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.decision_tree_sklearn import DecisionTreeRegressorSurrogate


@pytest.fixture
def sample_data():
    X = np.random.rand(30, 4)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.max(X, axis=1)
    ])
    return X, Y


def test_initialization_default():
    surrogate = DecisionTreeRegressorSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "DecisionTreeRegressor"


def test_initialization_with_params():
    surrogate = DecisionTreeRegressorSurrogate(max_depth=3)
    assert surrogate.model.estimator.max_depth == 3


def test_fit_predict_shape(sample_data):
    X, Y = sample_data
    surrogate = DecisionTreeRegressorSurrogate()
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_std_flag(sample_data):
    X, Y = sample_data
    surrogate = DecisionTreeRegressorSurrogate()
    surrogate.fit(X, Y)
    preds, std = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert std.shape == Y.shape
    assert np.allclose(std, 0)


def test_consistency_on_fit_predict(sample_data):
    X, Y = sample_data
    surrogate = DecisionTreeRegressorSurrogate()
    surrogate.fit(X, Y)
    preds1 = surrogate.predict(X)
    preds2 = surrogate.predict(X)
    np.testing.assert_allclose(preds1, preds2)
