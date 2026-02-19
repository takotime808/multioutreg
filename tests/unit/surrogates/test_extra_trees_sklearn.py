# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.extra_trees_sklearn import ExtraTreesRegressorSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(2026)
    X = np.random.rand(100, 4)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.std(X, axis=1),
    ])
    return X, Y


def test_initialization_default():
    surrogate = ExtraTreesRegressorSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "ExtraTreesRegressor"


def test_initialization_with_params():
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=10, max_depth=5)
    est = surrogate.model.estimator
    assert est.n_estimators == 10
    assert est.max_depth == 5


def test_fit_predict_output_shape(sample_data):
    X, Y = sample_data
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=10)
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_std(sample_data):
    X, Y = sample_data
    X_train, X_test = X[:80], X[80:]
    Y_train = Y[:80]
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=15, random_state=42)
    surrogate.fit(X_train, Y_train)
    preds, stds = surrogate.predict(X_test, return_std=True)
    assert preds.shape == (20, Y.shape[1])
    assert stds.shape == (20, Y.shape[1])
    assert (stds >= 0).all()
    assert not np.allclose(stds, 0.0)


def test_predict_consistency(sample_data):
    X, Y = sample_data
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=10, random_state=1)
    surrogate.fit(X, Y)
    preds1 = surrogate.predict(X)
    preds2 = surrogate.predict(X)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-6)


def test_std_computation_correctness(sample_data):
    X, Y = sample_data
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=5, random_state=0)
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)

    manual_stds = []
    for estimator in surrogate.model.estimators_:
        tree_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
        manual_stds.append(tree_preds.std(axis=0))
    manual_std = np.column_stack(manual_stds)

    np.testing.assert_allclose(stds, manual_std, rtol=1e-6)


def test_conformal_wrap(sample_data):
    X, Y = sample_data
    X_train, X_cal = X[:80], X[80:]
    Y_train, Y_cal = Y[:80], Y[80:]
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=10, random_state=0)
    surrogate.fit(X_train, Y_train)
    surrogate.wrap_conformal(X_cal, Y_cal)
    lower, upper = surrogate.conformal_predict(X_cal)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)
