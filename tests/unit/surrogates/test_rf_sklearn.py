# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(2025)
    X = np.random.rand(100, 4)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.std(X, axis=1)
    ])
    return X, Y


def test_initialization_default():
    surrogate = RandomForestSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "RandomForestRegressor"


def test_initialization_with_params():
    surrogate = RandomForestSurrogate(n_estimators=10, max_depth=5)
    est = surrogate.model.estimator
    assert est.n_estimators == 10
    assert est.max_depth == 5


def test_fit_predict_output_shape(sample_data):
    X, Y = sample_data
    surrogate = RandomForestSurrogate(n_estimators=10)
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_std(sample_data):
    X, Y = sample_data
    surrogate = RandomForestSurrogate(n_estimators=15, random_state=42)
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert (stds >= 0).all()
    assert not np.allclose(stds, 0.0)  # should not be all zero unless ensemble is identical


def test_predict_consistency(sample_data):
    X, Y = sample_data
    surrogate = RandomForestSurrogate(n_estimators=10, random_state=1)
    surrogate.fit(X, Y)
    preds1 = surrogate.predict(X)
    preds2 = surrogate.predict(X)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-6)


def test_std_computation_correctness(sample_data):
    X, Y = sample_data
    surrogate = RandomForestSurrogate(n_estimators=5, random_state=0)
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)

    # Manually compute std per output dimension
    manual_stds = []
    for output_idx, estimator in enumerate(surrogate.model.estimators_):
        # Get predictions for each tree in this output regressor
        tree_preds = np.array([tree.predict(X) for tree in estimator.estimators_])
        std_per_sample = tree_preds.std(axis=0)
        manual_stds.append(std_per_sample)
    manual_std = np.column_stack(manual_stds)

    np.testing.assert_allclose(stds, manual_std, rtol=1e-6)

