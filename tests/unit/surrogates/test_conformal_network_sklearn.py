# Copyright (c) 2026 takotime808

import numpy as np
import pytest

from multioutreg.surrogates.conformal_network_sklearn import ConformalPredictionNetworkSurrogate


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(42)
    X = rng.rand(120, 5)
    Y = np.column_stack([
        np.sin(2 * np.pi * X[:, 0]) + 0.1 * rng.randn(120),
        X[:, 1] ** 2 - X[:, 2] + 0.1 * rng.randn(120),
    ])
    return X, Y


def test_initialization_default():
    surrogate = ConformalPredictionNetworkSurrogate(random_state=0)
    assert surrogate.model.estimator.__class__.__name__ == "MLPRegressor"


def test_fit_predict_output_shape(sample_data):
    X, Y = sample_data
    surrogate = ConformalPredictionNetworkSurrogate(max_iter=200, random_state=0)
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape


def test_predict_with_std_returns_non_negative(sample_data):
    X, Y = sample_data
    surrogate = ConformalPredictionNetworkSurrogate(max_iter=200, random_state=0)
    surrogate.fit(X, Y)
    preds, std = surrogate.predict(X, return_std=True)

    assert preds.shape == Y.shape
    assert std.shape == Y.shape
    assert (std >= 0).all()
    assert not np.allclose(std, 0.0)


def test_invalid_conformal_params_raise():
    with pytest.raises(ValueError, match="calibration_fraction"):
        ConformalPredictionNetworkSurrogate(calibration_fraction=0.0)

    with pytest.raises(ValueError, match="conformal_quantile"):
        ConformalPredictionNetworkSurrogate(conformal_quantile=1.0)