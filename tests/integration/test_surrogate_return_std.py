# Copyright (c) 2026 takotime808
"""Every exported surrogate can predict, with ``return_std=True`` and ``return_std=False``.

``predict(X)``, ``predict(X, return_std=False)`` and ``predict(X, return_std=True)``
must all run after fitting, agree on the mean, and return the documented types:
a bare mean array without std, and a ``(mean, std)`` pair with it.
"""

import numpy as np
import pytest

from tests.unit.surrogates.test_all_surrogates_contract import (
    N_OUTPUTS,
    SURROGATE_NAMES,
    _build_and_fit,
    _skip_if_unavailable,
)

# Surrogates whose ``return_std=True`` mean comes from a different estimator than
# the deterministic ``return_std=False`` pass, so the two only agree approximately.
STOCHASTIC_MEAN = {
    "BNNSurrogate": "averages MC-dropout passes",
    "CatBoostSurrogate": "averages a virtual ensemble",
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, size=(80, 2))
    Y = np.column_stack([np.sin(3 * X[:, 0]) + X[:, 1], X[:, 0] * X[:, 1]])
    Y = Y + 0.05 * rng.normal(size=Y.shape)
    return X[:60], Y[:60], X[60:]


@pytest.fixture(scope="module")
def fitted():
    """Fit each surrogate once per module and share it across the tests below."""
    cache = {}

    def get(name, X, Y):
        if name not in cache:
            cache[name] = _build_and_fit(name, X, Y)
        return cache[name]

    return get


def _check_mean(mean, n_rows):
    assert isinstance(mean, np.ndarray), f"expected an ndarray, got {type(mean).__name__}"
    assert mean.shape == (n_rows, N_OUTPUTS)
    assert np.all(np.isfinite(mean))


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_predict_default(name, data, fitted):
    _skip_if_unavailable(name)
    X, Y, X_test = data
    _check_mean(fitted(name, X, Y).predict(X_test), len(X_test))


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_predict_return_std_false(name, data, fitted):
    _skip_if_unavailable(name)
    X, Y, X_test = data
    model = fitted(name, X, Y)
    mean = model.predict(X_test, return_std=False)
    assert not isinstance(mean, tuple), "return_std=False must return the mean alone"
    _check_mean(mean, len(X_test))
    np.testing.assert_allclose(mean, model.predict(X_test), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_predict_return_std_true(name, data, fitted):
    _skip_if_unavailable(name)
    X, Y, X_test = data
    model = fitted(name, X, Y)
    out = model.predict(X_test, return_std=True)
    assert isinstance(out, tuple) and len(out) == 2, "return_std=True must return (mean, std)"
    mean, std = out
    _check_mean(mean, len(X_test))
    std = np.asarray(std)
    assert std.shape == (len(X_test), N_OUTPUTS)
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)
    deterministic = model.predict(X_test, return_std=False)
    if name in STOCHASTIC_MEAN:
        # Same model, different estimator of its mean: stay within half a target std.
        assert np.all(np.abs(mean - deterministic) <= 0.5 * Y.std(axis=0)), STOCHASTIC_MEAN[name]
    else:
        np.testing.assert_allclose(mean, deterministic, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", SURROGATE_NAMES)
def test_predict_single_row(name, data, fitted):
    # A one-row query must keep its 2-D shape on both paths.
    _skip_if_unavailable(name)
    X, Y, X_test = data
    model = fitted(name, X, Y)
    _check_mean(model.predict(X_test[:1], return_std=False), 1)
    mean, std = model.predict(X_test[:1], return_std=True)
    _check_mean(mean, 1)
    assert np.asarray(std).shape == (1, N_OUTPUTS)
