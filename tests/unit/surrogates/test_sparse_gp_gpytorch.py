# Copyright (c) 2026 takotime808

import numpy as np
import pytest

pytest.importorskip("gpytorch", reason="gpytorch not installed")

from multioutreg.surrogates.sparse_gp_gpytorch import SparseGPSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(60, 3)
    Y = np.column_stack([
        np.sin(2 * np.pi * X[:, 0]) + 0.05 * np.random.randn(60),
        X[:, 1] - X[:, 2] + 0.05 * np.random.randn(60),
    ])
    return X, Y


class TestSparseGPSurrogate:
    def test_initialization_default(self):
        s = SparseGPSurrogate()
        assert s.n_inducing == 50
        assert s.max_iter == 100
        assert s.learning_rate == 0.1
        assert s.random_state is None

    def test_initialization_with_params(self):
        s = SparseGPSurrogate(n_inducing=20, max_iter=50, learning_rate=0.05, random_state=7)
        assert s.n_inducing == 20
        assert s.max_iter == 50
        assert s.learning_rate == 0.05
        assert s.random_state == 7

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=10, random_state=0)
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_fit_1d_y_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 3)).astype(np.float32)
        y = rng.standard_normal(40)
        s = SparseGPSurrogate(n_inducing=5, max_iter=5, random_state=0)
        s.fit(X, y)
        preds = s.predict(X)
        assert preds.shape == (40, 1)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=10, random_state=0)
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=10, random_state=0)
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_inducing_points_capped_at_n(self):
        """n_inducing > n_samples should be silently capped."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((20, 3))
        Y = rng.standard_normal((20, 1))
        s = SparseGPSurrogate(n_inducing=200, max_iter=5, random_state=0)
        s.fit(X, Y)  # should not raise
        preds = s.predict(X)
        assert preds.shape == (20, 1)

    def test_n_outputs_set_after_fit(self, sample_data):
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=5, random_state=0)
        s.fit(X, Y)
        assert s.n_outputs_ == Y.shape[1]

    def test_estimators_list_length(self, sample_data):
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=5, random_state=0)
        s.fit(X, Y)
        assert len(s.estimators_) == Y.shape[1]

    def test_input_normalisation_applied(self, sample_data):
        """Fitting should store _x_mean and _x_std."""
        X, Y = sample_data
        s = SparseGPSurrogate(n_inducing=10, max_iter=5, random_state=0)
        s.fit(X, Y)
        assert hasattr(s, "_x_mean")
        assert hasattr(s, "_x_std")
        assert s._x_mean.shape == (X.shape[1],)
        assert s._x_std.shape == (X.shape[1],)

    def test_predict_before_fit_raises(self):
        s = SparseGPSurrogate()
        with pytest.raises(AttributeError, match="not fitted"):
            s.predict(np.random.rand(5, 3))

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:40], X[40:]
        Y_train, Y_cal = Y[:40], Y[40:]
        s = SparseGPSurrogate(n_inducing=8, max_iter=10, random_state=0)
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_get_set_params(self):
        s = SparseGPSurrogate(n_inducing=30, max_iter=200)
        params = s.get_params()
        assert params["n_inducing"] == 30
        assert params["max_iter"] == 200
        s.set_params(n_inducing=15)
        assert s.n_inducing == 15

    def test_requires_gpytorch(self, monkeypatch):
        """SparseGPSurrogate raises ImportError when gpytorch is unavailable."""
        import multioutreg.surrogates.sparse_gp_gpytorch as mod
        monkeypatch.setattr(mod, "_GPYTORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="gpytorch"):
            SparseGPSurrogate()

    def test_random_state_reproducibility(self, sample_data):
        """Same random_state should give identical predictions."""
        X, Y = sample_data
        s1 = SparseGPSurrogate(n_inducing=10, max_iter=10, random_state=42)
        s2 = SparseGPSurrogate(n_inducing=10, max_iter=10, random_state=42)
        s1.fit(X, Y)
        s2.fit(X, Y)
        np.testing.assert_allclose(s1.predict(X), s2.predict(X), rtol=1e-4)
