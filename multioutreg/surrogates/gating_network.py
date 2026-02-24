# Copyright (c) 2026 takotime808

"""Gating network implementations for MixtureOfExpertsSurrogate."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class LinearGatingNetwork:
    """Linear softmax gating: input -> (n_experts,) expert weights.

    Implemented as multinomial logistic regression on hard-assignment labels.

    Parameters
    ----------
    n_experts : int
    random_state : int | None, default None
    """

    def __init__(self, n_experts: int, random_state: int | None = None):
        self.n_experts = n_experts
        self.random_state = random_state
        self._clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            random_state=random_state,
        )
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "LinearGatingNetwork":
        """Fit on hard-assignment labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        labels : np.ndarray, shape (n_samples,)  integer in [0, n_experts)
        """
        unique = np.unique(labels)
        if len(unique) < 2:
            # Degenerate: all samples assigned to one expert — skip classifier.
            self._degenerate_class = int(unique[0])
            self._scaler.fit(X)
            return self
        self._degenerate_class = None
        X_s = self._scaler.fit_transform(X)
        self._clf.fit(X_s, labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_experts) soft assignment weights."""
        if getattr(self, "_degenerate_class", None) is not None:
            out = np.zeros((X.shape[0], self.n_experts))
            out[:, self._degenerate_class] = 1.0
            return out
        X_s = self._scaler.transform(X)
        proba = self._clf.predict_proba(X_s)
        # Guard against degenerate case where some experts had no samples
        if proba.shape[1] < self.n_experts:
            full = np.zeros((X.shape[0], self.n_experts))
            for i, cls in enumerate(self._clf.classes_):
                full[:, cls] = proba[:, i]
            return full
        return proba


class MLPGatingNetwork:
    """MLP gating network with softmax output.

    Parameters
    ----------
    n_experts : int
    hidden_layer_sizes : tuple[int, ...], default (64, 32)
    random_state : int | None, default None
    """

    def __init__(
        self,
        n_experts: int,
        hidden_layer_sizes: tuple = (64, 32),
        random_state: int | None = None,
    ):
        self.n_experts = n_experts
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self._clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        )
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, labels: np.ndarray) -> "MLPGatingNetwork":
        """Fit on hard-assignment labels."""
        unique = np.unique(labels)
        if len(unique) < 2:
            self._degenerate_class = int(unique[0])
            self._scaler.fit(X)
            return self
        self._degenerate_class = None
        X_s = self._scaler.fit_transform(X)
        self._clf.fit(X_s, labels)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_experts) soft assignment weights."""
        if getattr(self, "_degenerate_class", None) is not None:
            out = np.zeros((X.shape[0], self.n_experts))
            out[:, self._degenerate_class] = 1.0
            return out
        X_s = self._scaler.transform(X)
        proba = self._clf.predict_proba(X_s)
        if proba.shape[1] < self.n_experts:
            full = np.zeros((X.shape[0], self.n_experts))
            for i, cls in enumerate(self._clf.classes_):
                full[:, cls] = proba[:, i]
            return full
        return proba
