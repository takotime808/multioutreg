# Copyright (c) 2026 takotime808

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")

from multioutreg.surrogates.deep_ensemble_pytorch import DeepEnsembleSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    Y = np.column_stack([
        3 * X[:, 0] + X[:, 1],
        np.mean(X, axis=1),
    ])
    return X, Y


class TestDeepEnsembleSurrogate:
    def test_initialization_default(self):
        s = DeepEnsembleSurrogate()
        assert s.n_estimators == 5
        assert s.hidden_layer_sizes == (128, 64)
        assert s.max_epochs == 500

    def test_initialization_with_params(self):
        s = DeepEnsembleSurrogate(
            n_estimators=3,
            hidden_layer_sizes=(32,),
            max_epochs=50,
            learning_rate=0.005,
            random_state=7,
        )
        assert s.n_estimators == 3
        assert s.hidden_layer_sizes == (32,)
        assert s.max_epochs == 50
        assert s.learning_rate == 0.005
        assert s.random_state == 7

    def test_multi_output_attribute(self):
        assert DeepEnsembleSurrogate._multi_output is True

    def test_fit_predict_shape(self, sample_data):
        X, Y = sample_data
        s = DeepEnsembleSurrogate(n_estimators=3, hidden_layer_sizes=(16,), max_epochs=5, random_state=0)
        s.fit(X, Y)
        preds = s.predict(X)
        assert preds.shape == Y.shape
        assert isinstance(preds, np.ndarray)

    def test_fit_1d_y_input(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 3))
        y = rng.standard_normal(60)
        s = DeepEnsembleSurrogate(n_estimators=2, hidden_layer_sizes=(8,), max_epochs=3, random_state=0)
        s.fit(X, y)
        preds = s.predict(X)
        assert preds.shape == (60, 1)

    def test_predict_return_std_shape(self, sample_data):
        X, Y = sample_data
        s = DeepEnsembleSurrogate(n_estimators=3, hidden_layer_sizes=(16,), max_epochs=5, random_state=0)
        s.fit(X, Y)
        preds, stds = s.predict(X, return_std=True)
        assert preds.shape == Y.shape
        assert stds.shape == Y.shape

    def test_predict_return_std_non_negative(self, sample_data):
        X, Y = sample_data
        s = DeepEnsembleSurrogate(n_estimators=3, hidden_layer_sizes=(16,), max_epochs=5, random_state=0)
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.all(stds >= 0)

    def test_predict_return_std_non_zero_with_multiple_estimators(self, sample_data):
        """With n_estimators > 1 and different seeds, std should be non-trivially zero."""
        X, Y = sample_data
        s = DeepEnsembleSurrogate(
            n_estimators=5, hidden_layer_sizes=(32, 16), max_epochs=10, random_state=0
        )
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        # At least some uncertainty should be non-zero across the ensemble
        assert not np.allclose(stds, 0.0)

    def test_n_estimators_1_gives_zero_std(self, sample_data):
        """Single member ensemble has no inter-model disagreement → std = 0."""
        X, Y = sample_data
        s = DeepEnsembleSurrogate(
            n_estimators=1, hidden_layer_sizes=(16,), max_epochs=5, random_state=0
        )
        s.fit(X, Y)
        _, stds = s.predict(X, return_std=True)
        assert np.allclose(stds, 0.0)

    def test_predict_before_fit_raises(self):
        s = DeepEnsembleSurrogate()
        with pytest.raises(AttributeError, match="not fitted"):
            s.predict(np.random.rand(5, 3))

    def test_conformal_wrap(self, sample_data):
        X, Y = sample_data
        X_train, X_cal = X[:70], X[70:]
        Y_train, Y_cal = Y[:70], Y[70:]
        s = DeepEnsembleSurrogate(n_estimators=3, hidden_layer_sizes=(16,), max_epochs=5, random_state=0)
        s.fit(X_train, Y_train)
        s.wrap_conformal(X_cal, Y_cal)
        lower, upper = s.conformal_predict(X_cal)
        assert lower.shape == Y_cal.shape
        assert upper.shape == Y_cal.shape
        assert np.all(upper >= lower)

    def test_networks_stored_after_fit(self, sample_data):
        X, Y = sample_data
        n = 3
        s = DeepEnsembleSurrogate(n_estimators=n, hidden_layer_sizes=(8,), max_epochs=3, random_state=0)
        s.fit(X, Y)
        assert hasattr(s, "networks_")
        assert len(s.networks_) == n

    def test_different_seeds_per_network(self, sample_data):
        """Each network in the ensemble should produce different predictions."""
        import torch
        X, Y = sample_data
        s = DeepEnsembleSurrogate(
            n_estimators=3, hidden_layer_sizes=(32, 16), max_epochs=5, random_state=0
        )
        s.fit(X, Y)
        X_t = torch.tensor(s.x_scaler_.transform(X), dtype=torch.float32)
        individual_preds = []
        for net in s.networks_:
            net.eval()
            with torch.no_grad():
                individual_preds.append(net(X_t).numpy())
        # Networks should not all be identical
        assert not np.allclose(individual_preds[0], individual_preds[1])

    def test_get_set_params(self):
        s = DeepEnsembleSurrogate(n_estimators=4, max_epochs=200)
        params = s.get_params()
        assert params["n_estimators"] == 4
        assert params["max_epochs"] == 200
        s.set_params(n_estimators=2)
        assert s.n_estimators == 2

    def test_requires_torch(self, monkeypatch):
        """DeepEnsembleSurrogate raises ImportError when torch is unavailable."""
        import multioutreg.surrogates._torch_utils as utils
        monkeypatch.setattr(utils, "_TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch"):
            DeepEnsembleSurrogate()
