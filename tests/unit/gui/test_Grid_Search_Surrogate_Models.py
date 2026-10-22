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

def test_gradient_boosting_with_uncertainty_std_is_sigma():
    # Regression: std was the half-width of the 95% quantile interval
    # (~1.96 * sigma for Gaussian noise) instead of sigma itself.
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(2000, 1))
    y = X[:, 0] + rng.normal(size=2000)  # sigma = 1
    _, std = GradientBoostingWithUncertainty().fit(X, y).predict(X[:500], return_std=True)
    assert abs(std.mean() - 1.0) < 0.2


def test_forest_wrappers_add_noise_variance():
    # Audit item 27: tree spread alone is epistemic only; the wrappers add a
    # held-out noise variance per output. ExtraTrees (no bootstrap) uses K-fold
    # refits, which must not recurse into the wrapper's own fit.
    from multioutreg.gui.grid_search_core import ExtraTreesWithUncertainty

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 3))
    y = np.sin(6 * X[:, 0]) + 0.3 * rng.normal(size=200)
    for cls in (RandomForestWithUncertainty, ExtraTreesWithUncertainty):
        model = cls(n_estimators=30, random_state=0).fit(X, y)
        assert np.isscalar(model.noise_var_) and model.noise_var_ >= 0
        mean, std = model.predict(X, return_std=True)
        tree_var = np.stack([t.predict(X) for t in model.estimators_]).var(axis=0)
        np.testing.assert_allclose(std, np.sqrt(tree_var + model.noise_var_))
        multi = cls(n_estimators=30, random_state=0).fit(X, np.column_stack([y, 2 * y]))
        assert multi.noise_var_.shape == (2,)
