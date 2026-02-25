# Copyright (c) 2026 takotime808

import numpy as np
import pytest

pytest.importorskip("catboost", reason="catboost not installed")

from multioutreg.surrogates.catboost_sklearn import CatBoostSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(80, 5)
    Y = np.column_stack([
        3 * X[:, 0] + X[:, 1],
        np.mean(X, axis=1),
    ])
    return X, Y


class TestCatBoostSurrogate:
    def test_initialization_default(self):
        s = CatBoostSurrogate()
        assert s.n_estimators == 200
        assert s.learning_rate == 0.05
        assert s.use_uncertainty is True

    def test_initialization_with_params(self):
        s = CatBoostSurrogate(n_estimators=50, learning_rate=0.01, use_uncertainty=False)
        assert s.n_estimators == 50
        assert s.learning_rate == 0.01
        assert s.use_uncertainty is False

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=False, random_seed=0)
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=True, random_seed=0)
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=True, random_seed=0)
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_predict_return_std_non_zero_with_uncertainty(self, sample_data):
        """Virtual ensemble should produce non-trivial uncertainty estimates."""
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=True, random_seed=0)
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert not np.allclose(stds, 0.0)

    def test_predict_return_std_false_uncertainty(self, sample_data):
        """Without use_uncertainty, return_std falls back to zeros."""
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=False, random_seed=0)
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:60], X[60:]
        Y_train, Y_cal = Y[:60], Y[60:]
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=False, random_seed=0)
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_get_set_params(self):
        s = CatBoostSurrogate(n_estimators=50, learning_rate=0.05)
        params = s.get_params()
        assert params["n_estimators"] == 50
        assert params["learning_rate"] == 0.05
        s.set_params(n_estimators=100)
        assert s.n_estimators == 100

    def test_predict_consistency(self, sample_data):
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=False, random_seed=0)
        s.fit(X, Y)
        np.testing.assert_allclose(s.predict(X), s.predict(X), rtol=1e-5)

    def test_rmse_with_uncertainty_raw_output_shape(self, sample_data):
        """CatBoostRegressor trained with RMSEWithUncertainty returns (n_samples, 2)
        from .predict() — column 0 is mean, column 1 is variance.  The surrogate's
        virtual_ensembles_predict path unpacks this correctly; calling the raw
        MultiOutputRegressor.predict() directly would produce the wrong shape.
        This test documents and verifies that contract end-to-end.
        """
        X, Y = sample_data
        s = CatBoostSurrogate(n_estimators=20, use_uncertainty=True, random_seed=0)
        s.fit(X, Y)

        # Raw per-output model returns (n_samples, 2): [mean, variance]
        raw = s.model.estimators_[0].predict(X)
        assert raw.ndim == 2, "RMSEWithUncertainty predict() must return 2-D array"
        assert raw.shape == (X.shape[0], 2), f"Expected ({X.shape[0]}, 2), got {raw.shape}"

        # Column 0 = mean prediction, column 1 = variance (must be non-negative)
        means = raw[:, 0]
        variances = raw[:, 1]
        assert np.all(variances >= 0), "Variance column must be non-negative"

        # The surrogate's predict(return_std=True) path uses virtual_ensembles_predict,
        # which also returns (n_samples, 3): [mean, total_var, knowledge_var].
        # Verify that the surrogate correctly extracts mean and sqrt(total_var).
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape
        assert np.all(stds >= 0)

        # Manually reproduce the virtual_ensembles_predict extraction for output 0
        unc = s.model.estimators_[0].virtual_ensembles_predict(
            X,
            virtual_ensembles_count=s.virtual_ensembles_count,
            prediction_type="TotalUncertainty",
        )
        assert unc.shape[1] == 3, "TotalUncertainty must return 3 columns"
        expected_pred_0 = unc[:, 0]
        expected_std_0 = np.sqrt(np.maximum(unc[:, 1], 0.0))
        np.testing.assert_allclose(preds[:, 0], expected_pred_0, rtol=1e-5)
        np.testing.assert_allclose(stds[:, 0], expected_std_0, rtol=1e-5)

    def test_requires_catboost(self, monkeypatch):
        """CatBoostSurrogate raises ImportError when catboost is unavailable."""
        import multioutreg.surrogates.catboost_sklearn as mod
        monkeypatch.setattr(mod, "_CATBOOST_AVAILABLE", False)
        with pytest.raises(ImportError, match="catboost"):
            CatBoostSurrogate()
