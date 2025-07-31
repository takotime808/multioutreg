# Copyright (c) 2025 takotime808

import numpy as np
import pytest

from multioutreg.surrogates import (
    MultiFidelitySurrogate,
    LinearRegressionSurrogate,
)


@pytest.fixture
def sample_data():
    X_low = np.random.rand(10, 3)
    Y_low = np.column_stack([
        np.sum(X_low, axis=1),
        np.prod(X_low, axis=1),
    ])
    X_high = np.random.rand(8, 3)
    Y_high = np.column_stack([
        np.sum(X_high, axis=1),
        np.prod(X_high, axis=1),
    ])
    return {
        "low": (X_low, Y_low),
        "high": (X_high, Y_high),
    }


def test_fit_and_predict(sample_data):
    mfs = MultiFidelitySurrogate(LinearRegressionSurrogate, ["low", "high"])
    mfs.fit(sample_data)
    X_low, Y_low = sample_data["low"]
    X_high, Y_high = sample_data["high"]
    preds_low = mfs.predict("low", X_low)
    preds_high = mfs.predict("high", X_high)
    assert preds_low.shape == Y_low.shape
    assert preds_high.shape == Y_high.shape


def test_invalid_level(sample_data):
    mfs = MultiFidelitySurrogate(LinearRegressionSurrogate, ["low"])
    mfs.fit({"low": sample_data["low"]})
    with pytest.raises(ValueError):
        mfs.predict("high", sample_data["high"][0])