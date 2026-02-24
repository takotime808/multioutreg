# Copyright (c) 2026 takotime808

import numpy as np
import pytest

pytest.importorskip("ngboost", reason="ngboost not installed")

from multioutreg.surrogates.ngboost_sklearn import NGBoostSurrogate


@pytest.fixture
def sample_data():
    np.random.seed(2026)
    X = np.random.rand(80, 4)
    Y = np.column_stack([
        np.sum(X, axis=1),
        np.std(X, axis=1),
    ])
    return X, Y


def test_initialization_default():
    surrogate = NGBoostSurrogate()
    assert surrogate.model.estimator.__class__.__name__ == "NGBRegressor"


def test_initialization_with_params():
    surrogate = NGBoostSurrogate(n_estimators=50, learning_rate=0.05)
    est = surrogate.model.estimator
    assert est.n_estimators == 50
    assert est.learning_rate == 0.05


def test_fit_predict_output_shape(sample_data):
    X, Y = sample_data
    surrogate = NGBoostSurrogate(n_estimators=50, verbose=False)
    surrogate.fit(X, Y)
    preds = surrogate.predict(X)
    assert preds.shape == Y.shape
    assert isinstance(preds, np.ndarray)


def test_predict_with_std(sample_data):
    X, Y = sample_data
    surrogate = NGBoostSurrogate(n_estimators=50, verbose=False)
    surrogate.fit(X, Y)
    preds, stds = surrogate.predict(X, return_std=True)
    assert preds.shape == Y.shape
    assert stds.shape == Y.shape
    assert np.all(stds > 0)


def test_conformal_wrap(sample_data):
    X, Y = sample_data
    X_train, X_cal = X[:60], X[60:]
    Y_train, Y_cal = Y[:60], Y[60:]
    surrogate = NGBoostSurrogate(n_estimators=50, verbose=False)
    surrogate.fit(X_train, Y_train)
    surrogate.wrap_conformal(X_cal, Y_cal)
    lower, upper = surrogate.conformal_predict(X_cal)
    assert lower.shape == Y_cal.shape
    assert upper.shape == Y_cal.shape
    assert np.all(upper >= lower)


def test_requires_ngboost_import(monkeypatch):
    """NGBoostSurrogate raises ImportError when ngboost is not available."""
    import multioutreg.surrogates.ngboost_sklearn as mod
    monkeypatch.setattr(mod, "_NGBOOST_AVAILABLE", False)
    with pytest.raises(ImportError, match="ngboost"):
        NGBoostSurrogate()
