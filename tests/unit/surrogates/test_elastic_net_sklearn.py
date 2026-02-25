# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates.elastic_net_sklearn import ElasticNetSurrogate, LassoSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(80, 5)
    Y = np.column_stack([
        3 * X[:, 0] + 2 * X[:, 1] - X[:, 2],
        np.mean(X, axis=1),
    ])
    return X, Y


# ── ElasticNetSurrogate ────────────────────────────────────────────────────────

class TestElasticNetSurrogate:
    def test_initialization_default(self):
        s = ElasticNetSurrogate()
        assert s.alpha == 1.0
        assert s.l1_ratio == 0.5

    def test_initialization_with_params(self):
        s = ElasticNetSurrogate(alpha=0.1, l1_ratio=0.8, max_iter=500)
        assert s.alpha == 0.1
        assert s.l1_ratio == 0.8
        assert s.max_iter == 500

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = ElasticNetSurrogate()
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = ElasticNetSurrogate()
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = ElasticNetSurrogate()
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_predict_consistency(self, sample_data):
        X, Y = sample_data
        s = ElasticNetSurrogate()
        s.fit(X, Y)
        np.testing.assert_allclose(s.predict(X), s.predict(X), rtol=1e-6)

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:60], X[60:]
        Y_train, Y_cal = Y[:60], Y[60:]
        s = ElasticNetSurrogate()
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_get_set_params(self):
        s = ElasticNetSurrogate(alpha=0.5, l1_ratio=0.3)
        params = s.get_params()
        assert params["alpha"] == 0.5
        assert params["l1_ratio"] == 0.3
        s.set_params(alpha=0.2)
        assert s.alpha == 0.2

    def test_l1_ratio_1_behaves_like_lasso(self, sample_data):
        """ElasticNet with l1_ratio=1.0 is equivalent to Lasso."""
        X, Y = sample_data
        s = ElasticNetSurrogate(alpha=0.1, l1_ratio=1.0)
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape

    def test_l1_ratio_0_behaves_like_ridge(self, sample_data):
        """ElasticNet with l1_ratio=0.0 is purely L2 (Ridge)."""
        X, Y = sample_data
        s = ElasticNetSurrogate(alpha=0.1, l1_ratio=0.0)
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape

    def test_sparse_coefficients_high_alpha(self, sample_data):
        """High alpha should push many coefficients to zero."""
        X, Y = sample_data
        s = ElasticNetSurrogate(alpha=10.0, l1_ratio=0.9)
        s.fit(X, Y)
        # At least one output model should have sparse coefs
        estimators = s.model.estimators_
        assert len(estimators) == Y.shape[1]


# ── LassoSurrogate ─────────────────────────────────────────────────────────────

class TestLassoSurrogate:
    def test_initialization_default(self):
        s = LassoSurrogate()
        assert s.alpha == 1.0

    def test_initialization_with_params(self):
        s = LassoSurrogate(alpha=0.05, max_iter=2000)
        assert s.alpha == 0.05
        assert s.max_iter == 2000

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = LassoSurrogate()
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = LassoSurrogate()
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = LassoSurrogate()
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_predict_consistency(self, sample_data):
        X, Y = sample_data
        s = LassoSurrogate()
        s.fit(X, Y)
        np.testing.assert_allclose(s.predict(X), s.predict(X), rtol=1e-6)

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:60], X[60:]
        Y_train, Y_cal = Y[:60], Y[60:]
        s = LassoSurrogate()
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_get_set_params(self):
        s = LassoSurrogate(alpha=0.3)
        params = s.get_params()
        assert params["alpha"] == 0.3
        s.set_params(alpha=0.7)
        assert s.alpha == 0.7

    def test_high_alpha_produces_zero_predictions(self):
        """Very high alpha forces all coefs to zero → predictions near intercept."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 10))
        y = rng.standard_normal((100, 1))
        s = LassoSurrogate(alpha=1e6)
        s.fit(X, y)
        preds = s.predict(X)
        # With huge alpha all coefficients are zeroed out
        assert np.allclose(preds, preds[0], atol=1e-3)
