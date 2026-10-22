# Copyright (c) 2026 takotime808

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")

from multioutreg.surrogates.bnn_pytorch import BNNSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(80, 5)
    Y = np.column_stack([3 * X[:, 0] + X[:, 1], np.mean(X, axis=1)])
    return X, Y


def test_fit_and_predict(sample_data):
    X, Y = sample_data
    m = BNNSurrogate(hidden_layer_sizes=(32, 16), max_epochs=20, random_state=0)
    m.fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_return_std_flag(sample_data):
    X, Y = sample_data
    m = BNNSurrogate(
        hidden_layer_sizes=(32, 16), max_epochs=20, n_mc_samples=10, random_state=0
    )
    m.fit(X, Y)
    preds, stds = m.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.all(stds >= 0)
    # std should be non-zero (dropout produces variation)
    assert not np.allclose(stds, 0.0)


def test_mc_samples_1_gives_zero_std(sample_data):
    X, Y = sample_data
    m = BNNSurrogate(
        hidden_layer_sizes=(32, 16), max_epochs=20, n_mc_samples=1, random_state=0,
        loss="mse",
    )
    m.fit(X, Y)
    preds, stds = m.predict(X, return_std=True)
    assert np.allclose(stds, 0.0)


def test_multi_output_attribute():
    assert BNNSurrogate._multi_output is True


def test_conformal_wrap(sample_data):
    X, Y = sample_data
    X_train, X_cal = X[:60], X[60:]
    Y_train, Y_cal = Y[:60], Y[60:]
    m = BNNSurrogate(hidden_layer_sizes=(32, 16), max_epochs=20, random_state=0)
    m.fit(X_train, Y_train)
    m.wrap_conformal(X_cal, Y_cal)
    lower, upper = m.conformal_predict(X_cal)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)


def test_predict_before_fit_raises():
    m = BNNSurrogate()
    with pytest.raises(AttributeError, match="not fitted"):
        m.predict(np.random.rand(5, 3))


def test_training_losses_decrease(sample_data):
    X, Y = sample_data
    m = BNNSurrogate(
        hidden_layer_sizes=(32, 16), max_epochs=50, patience=50, random_state=0
    )
    m.fit(X, Y)
    # The loss should generally decrease over training
    assert m.training_losses_[0] > m.training_losses_[-1]


def test_mc_samples_1_nll_gives_noise_std(sample_data):
    """With the NLL head a single pass still returns the learned noise std."""
    X, Y = sample_data
    m = BNNSurrogate(
        hidden_layer_sizes=(32, 16), max_epochs=20, n_mc_samples=1, random_state=0
    )
    m.fit(X, Y)
    _, stds = m.predict(X, return_std=True)
    assert np.all(stds > 0)


def test_invalid_loss_raises():
    with pytest.raises(ValueError):
        BNNSurrogate(loss="huber")


def test_nll_head_learns_noise_and_covers():
    """MC spread alone under-covers on noisy data; the NLL head adds the noise."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (1000, 2))
    noise = 0.1 + 0.5 * X[:, 0]
    y = (np.sin(6 * X[:, 0]) + X[:, 1] + noise * rng.standard_normal(1000)).reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = X[:400], X[400:], y[:400], y[400:]
    cover = {}
    for loss in ("mse", "nll"):
        m = BNNSurrogate(hidden_layer_sizes=(64, 32), max_epochs=300, random_state=0, loss=loss)
        pred, std = m.fit(X_tr, y_tr).predict(X_te, return_std=True)
        cover[loss] = np.mean(np.abs(y_te - pred) <= 1.645 * std)
        if loss == "nll":
            # heteroscedastic: std grows with the noise level
            assert np.corrcoef(std[:, 0], noise[400:])[0, 1] > 0.7
    assert cover["mse"] < 0.7
    assert 0.8 <= cover["nll"] <= 0.99
