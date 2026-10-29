# Copyright (c) 2026 takotime808

"""Gating networks used by MixtureOfExpertsSurrogate."""

import numpy as np
import pytest

from multioutreg.surrogates.gating_network import LinearGatingNetwork, MLPGatingNetwork

GATES = [
    pytest.param(lambda n: LinearGatingNetwork(n, random_state=0), id="linear"),
    pytest.param(lambda n: MLPGatingNetwork(n, hidden_layer_sizes=(8,), random_state=0), id="mlp"),
]


@pytest.fixture
def X():
    return np.linspace(-3, 3, 60).reshape(-1, 1)


@pytest.mark.parametrize("make", GATES)
def test_separable_labels_are_learned(make, X):
    labels = (X[:, 0] > 0).astype(int)
    proba = make(2).fit(X, labels).predict_proba(X)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.mean(proba.argmax(axis=1) == labels) > 0.9


@pytest.mark.parametrize("make", GATES)
def test_single_expert_assignment_is_one_hot(make, X):
    gate = make(3).fit(X, np.full(60, 1))
    proba = gate.predict_proba(X[:5])
    np.testing.assert_array_equal(proba, np.tile([0.0, 1.0, 0.0], (5, 1)))


@pytest.mark.parametrize("make", GATES)
def test_unused_expert_gets_zero_weight(make, X):
    # Only experts 0 and 2 receive samples; the output still has 3 columns.
    labels = np.where(X[:, 0] > 0, 2, 0)
    proba = make(3).fit(X, labels).predict_proba(X)
    assert proba.shape == (60, 3)
    np.testing.assert_array_equal(proba[:, 1], 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba[-1, 2] > 0.5 and proba[0, 0] > 0.5
