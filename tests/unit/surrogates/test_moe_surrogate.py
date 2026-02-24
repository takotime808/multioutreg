# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate
from multioutreg.surrogates.linear_sklearn import LinearRegressionSurrogate


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 4))
    Y = np.column_stack([
        3 * X[:, 0] + X[:, 1],
        np.sum(X, axis=1),
    ])
    return X, Y


@pytest.fixture
def multimodal_data():
    """Two distinct linear regimes separated in input space."""
    rng = np.random.default_rng(0)
    n = 120
    X = rng.standard_normal((n, 3))
    # Region A: x[:,0] > 0
    Y = np.zeros((n, 2))
    mask = X[:, 0] > 0
    Y[mask, 0] = 10 * X[mask, 0] + 1
    Y[mask, 1] = 5 * X[mask, 1]
    Y[~mask, 0] = -2 * X[~mask, 0] - 5
    Y[~mask, 1] = -3 * X[~mask, 1] + 2
    return X, Y


def test_fit_and_predict_shape(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(n_experts=3, max_em_iters=5, random_state=0)
    m.fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == Y.shape


def test_predict_with_return_std(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(n_experts=2, max_em_iters=5, random_state=0)
    m.fit(X, Y)
    preds, stds = m.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.all(stds >= 0)


def test_multi_output_attribute():
    assert MixtureOfExpertsSurrogate._multi_output is True


def test_routing_weights_shape(sample_data):
    X, Y = sample_data
    n_experts = 3
    m = MixtureOfExpertsSurrogate(n_experts=n_experts, max_em_iters=5, random_state=0)
    m.fit(X, Y)
    weights = m.get_routing_weights(X)
    assert weights.shape == (X.shape[0], n_experts)


def test_routing_weights_sum_to_one(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(n_experts=4, max_em_iters=5, random_state=0)
    m.fit(X, Y)
    weights = m.get_routing_weights(X)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)


def test_hard_routing(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(
        n_experts=2, routing="hard", max_em_iters=5, random_state=0
    )
    m.fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == Y.shape


def test_1d_y_input():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 3))
    y = rng.standard_normal(60)
    m = MixtureOfExpertsSurrogate(n_experts=2, max_em_iters=3, random_state=0)
    m.fit(X, y)
    preds = m.predict(X)
    assert preds.shape == (60, 1)


def test_mlp_gating_type(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(
        n_experts=2, gating_type="mlp", max_em_iters=3, random_state=0
    )
    m.fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == Y.shape


def test_expert_specialization(multimodal_data):
    """Gating weights should be non-uniform on multi-modal data."""
    X, Y = multimodal_data
    m = MixtureOfExpertsSurrogate(n_experts=2, max_em_iters=15, random_state=0)
    m.fit(X, Y)
    weights = m.get_routing_weights(X)
    # At least one sample should clearly prefer expert 0 over expert 1
    assert weights[:, 0].max() > 0.7 or weights[:, 1].max() > 0.7


def test_heterogeneous_experts(sample_data):
    X, Y = sample_data
    m = MixtureOfExpertsSurrogate(
        n_experts=2,
        expert_type=[RandomForestSurrogate, LinearRegressionSurrogate],
        max_em_iters=3,
        random_state=0,
    )
    m.fit(X, Y)
    preds = m.predict(X)
    assert preds.shape == Y.shape


def test_conformal_wrap(sample_data):
    X, Y = sample_data
    X_train, X_cal = X[:70], X[70:]
    Y_train, Y_cal = Y[:70], Y[70:]
    m = MixtureOfExpertsSurrogate(n_experts=2, max_em_iters=5, random_state=0)
    m.fit(X_train, Y_train)
    m.wrap_conformal(X_cal, Y_cal)
    lower, upper = m.conformal_predict(X_cal)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)


def test_predict_before_fit_raises():
    m = MixtureOfExpertsSurrogate()
    with pytest.raises(AttributeError, match="not fitted"):
        m.predict(np.random.rand(5, 3))


def test_routing_weights_before_fit_raises():
    m = MixtureOfExpertsSurrogate()
    with pytest.raises(AttributeError, match="not fitted"):
        m.get_routing_weights(np.random.rand(5, 3))


def test_invalid_gating_type_raises():
    m = MixtureOfExpertsSurrogate(gating_type="invalid")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 2))
    Y = rng.standard_normal((20, 1))
    with pytest.raises(ValueError, match="Unknown gating_type"):
        m.fit(X, Y)


def test_get_set_params():
    m = MixtureOfExpertsSurrogate(n_experts=3, gating_type="mlp", random_state=1)
    params = m.get_params()
    assert params["n_experts"] == 3
    assert params["gating_type"] == "mlp"
    m.set_params(n_experts=5)
    assert m.n_experts == 5
