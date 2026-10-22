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
    surrogate = ExtraTreesRegressorSurrogate(n_estimators=5, random_state=0, include_noise=False)
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


def test_std_adds_noise_variance(sample_data):
    X, Y = sample_data
    epistemic = ExtraTreesRegressorSurrogate(n_estimators=20, random_state=0, include_noise=False).fit(X, Y)
    total = ExtraTreesRegressorSurrogate(n_estimators=20, random_state=0).fit(X, Y)
    _, s_tree = epistemic.predict(X, return_std=True)
    _, s_total = total.predict(X, return_std=True)
    assert total.noise_var_.shape == (Y.shape[1],)
    assert np.all(total.noise_var_ >= 0)
    np.testing.assert_allclose(s_total, np.sqrt(s_tree ** 2 + total.noise_var_), rtol=1e-6)


def test_noisy_data_interval_coverage():
    """Tree spread alone under-covers on noisy data; with the extra term ~90%."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (700, 3))
    y = np.sin(6 * X[:, 0]) + X[:, 1] + 0.3 * rng.standard_normal(700)
    Y = np.column_stack([y, y])
    X_tr, X_te, Y_tr, Y_te = X[:300], X[300:], Y[:300], Y[300:]
    m = ExtraTreesRegressorSurrogate(n_estimators=100, random_state=0).fit(X_tr, Y_tr)
    pred, std = m.predict(X_te, return_std=True)
    coverage = np.mean(np.abs(Y_te - pred) <= 1.645 * std)
    assert 0.83 <= coverage <= 0.97
    _, std_tree = ExtraTreesRegressorSurrogate(n_estimators=100, random_state=0, include_noise=False).fit(
        X_tr, Y_tr).predict(X_te, return_std=True)
    assert np.mean(np.abs(Y_te - pred) <= 1.645 * std_tree) <= coverage
