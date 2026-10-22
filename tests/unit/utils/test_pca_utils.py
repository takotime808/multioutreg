# Copyright (c) 2026 takotime808

import numpy as np
import pytest
from sklearn.decomposition import PCA

from multioutreg.utils.pca_utils import fit_input_pca


def _two_factor_data(n=300, seed=0):
    """Six inputs driven by two latent factors (3 inputs each) plus small noise."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 2))
    X = np.repeat(f, 3, axis=1) + 0.3 * rng.standard_normal((n, 6))
    return X[:240], X[240:]


def test_kaiser_finds_latent_factors_regardless_of_units():
    X_train, X_test = _two_factor_data()
    # Inputs in tiny units: raw-data eigenvalues are all < 1, so the old
    # unscaled Kaiser rule would have suggested 0 components.
    small = 0.01
    assert np.sum(PCA().fit(X_train * small).explained_variance_ > 1) == 0
    res = fit_input_pca(X_train * small, X_test * small)
    assert res.kaiser_k == 2
    assert res.n_components == 2


def test_result_is_invariant_to_per_column_scale():
    X_train, X_test = _two_factor_data()
    scales = np.array([1e-3, 1.0, 1e3, 5.0, 0.2, 1e4])
    a = fit_input_pca(X_train, X_test)
    b = fit_input_pca(X_train * scales, X_test * scales)
    assert a.kaiser_k == b.kaiser_k
    # Components match up to sign.
    np.testing.assert_allclose(np.abs(a.X_test), np.abs(b.X_test), atol=1e-8)


def test_scaler_and_pca_fit_on_train_only():
    X_train, X_test = _two_factor_data()
    a = fit_input_pca(X_train, X_test)
    b = fit_input_pca(X_train, X_test * 100 + 7)
    np.testing.assert_array_equal(a.X_train, b.X_train)
    scaler = a.transformer.named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))


def test_threshold_picks_smallest_k_reaching_target():
    X_train, X_test = _two_factor_data()
    res = fit_input_pca(X_train, X_test, "Explained variance threshold", threshold=0.8)
    cum = np.cumsum(res.preview_pca.explained_variance_ratio_)
    assert cum[res.n_components - 1] >= 0.8
    assert res.n_components == 1 or cum[res.n_components - 2] < 0.8
    assert res.X_train.shape[1] == res.n_components
    assert res.feature_names == [f"PC{i + 1}" for i in range(res.n_components)]


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("Manual", {"n_components": 50}),
        ("Manual", {"n_components": 0}),
        ("Explained variance threshold", {"threshold": 1.0 + 1e-9}),
    ],
)
def test_n_components_clipped_to_valid_range(method, kwargs):
    X_train, X_test = _two_factor_data()
    res = fit_input_pca(X_train, X_test, method, **kwargs)
    assert 1 <= res.n_components <= X_train.shape[1]


def test_manual_without_n_components_falls_back_to_kaiser():
    X_train, X_test = _two_factor_data()
    res = fit_input_pca(X_train, X_test, "Manual", n_components=None)
    assert res.n_components == res.kaiser_k
