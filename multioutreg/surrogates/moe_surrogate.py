# Copyright (c) 2026 takotime808

"""Mixture of Experts surrogate model."""

from __future__ import annotations

from typing import Literal

import numpy as np

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class MixtureOfExpertsSurrogate(ConformalMixin):
    """Mixture of Experts surrogate with learned gating network.

    Specializes K expert regressors to different regions of input space
    using a gating network trained via a hard-EM loop.

    Parameters
    ----------
    n_experts : int, default 4
    expert_type : type | list[type] | None
        Surrogate class(es) to use as experts.  A single type is used for all
        experts; a list of types enables heterogeneous experts (one per slot).
        Defaults to ``RandomForestSurrogate``.
    expert_params : dict | list[dict] | None
        Constructor keyword arguments for each expert.  A single dict applies
        to all experts; a list is paired element-wise with ``expert_type``.
    gating_type : {"linear", "mlp"}, default "linear"
    routing : {"soft", "hard"}, default "soft"
        ``"soft"``: predictions are a weighted sum over all experts.
        ``"hard"``: each sample is routed to the highest-weight expert only.
    max_em_iters : int, default 20
        Maximum number of hard-EM iterations.  Stops early if assignments
        do not change between iterations.
    random_state : int | None, default None
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(
        self,
        n_experts: int = 4,
        expert_type=None,
        expert_params: dict | list[dict] | None = None,
        gating_type: Literal["linear", "mlp"] = "linear",
        routing: Literal["soft", "hard"] = "soft",
        max_em_iters: int = 20,
        random_state: int | None = None,
    ):
        self.n_experts = n_experts
        self.expert_type = expert_type
        self.expert_params = expert_params
        self.gating_type = gating_type
        self.routing = routing
        self.max_em_iters = max_em_iters
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_expert(self, k: int):
        from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate

        if self.expert_type is None:
            cls = RandomForestSurrogate
        elif isinstance(self.expert_type, list):
            cls = self.expert_type[k]
        else:
            cls = self.expert_type

        if self.expert_params is None:
            params: dict = {}
        elif isinstance(self.expert_params, list):
            params = dict(self.expert_params[k])
        else:
            params = dict(self.expert_params)

        return cls(**params)

    def _make_gating_network(self):
        from multioutreg.surrogates.gating_network import (
            LinearGatingNetwork,
            MLPGatingNetwork,
        )

        if self.gating_type == "linear":
            return LinearGatingNetwork(self.n_experts, random_state=self.random_state)
        elif self.gating_type == "mlp":
            return MLPGatingNetwork(self.n_experts, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown gating_type: {self.gating_type!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MixtureOfExpertsSurrogate":
        """Fit experts and gating network using a hard-EM loop.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        Y : np.ndarray, shape (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
        """
        rng = np.random.default_rng(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples = X.shape[0]
        self.n_outputs_ = Y.shape[1]

        # Initialise with random hard assignments; ensure every expert gets ≥1 sample
        labels = rng.integers(0, self.n_experts, size=n_samples)
        for k in range(self.n_experts):
            if np.sum(labels == k) == 0:
                labels[rng.integers(0, n_samples)] = k

        experts = None
        gating = None
        for _iter in range(self.max_em_iters):
            # M-step: fit each expert on its currently assigned samples
            experts = []
            for k in range(self.n_experts):
                mask = labels == k
                if mask.sum() == 0:
                    mask = np.ones(n_samples, dtype=bool)
                expert = self._make_expert(k)
                expert.fit(X[mask], Y[mask])
                experts.append(expert)

            # Fit gating network on current hard assignments
            gating = self._make_gating_network()
            gating.fit(X, labels)

            # E-step: compute responsibilities and derive new hard assignments
            responsibilities = gating.predict_proba(X)  # (n_samples, n_experts)
            new_labels = responsibilities.argmax(axis=1)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

        self.experts_ = experts
        self.gating_ = gating
        return self

    def get_routing_weights(self, X: np.ndarray) -> np.ndarray:
        """Return gating weights for each sample.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        weights : np.ndarray, shape (n_samples, n_experts)
        """
        if not hasattr(self, "gating_"):
            raise AttributeError("MixtureOfExpertsSurrogate is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        return self.gating_.predict_proba(X)

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict outputs for X.

        Parameters
        ----------
        X : np.ndarray
        return_std : bool, default False
            If True, also return uncertainty estimated from expert disagreement.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std : np.ndarray, shape (n_samples, n_outputs)  [only if return_std=True]
        """
        if not hasattr(self, "experts_"):
            raise AttributeError("MixtureOfExpertsSurrogate is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        weights = self.get_routing_weights(X)  # (n_samples, n_experts)

        expert_preds = []
        for expert in self.experts_:
            pred = np.asarray(expert.predict(X))
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            expert_preds.append(pred)

        # Stack to (n_experts, n_samples, n_outputs)
        expert_preds = np.stack(expert_preds, axis=0)

        if self.routing == "hard":
            assignment = weights.argmax(axis=1)  # (n_samples,)
            y_pred = expert_preds[assignment, np.arange(X.shape[0]), :]
            if return_std:
                return y_pred, np.zeros_like(y_pred)
            return y_pred

        # Soft routing: weighted sum over experts
        # w shape: (n_experts, n_samples, 1)
        w = weights.T[:, :, np.newaxis]
        y_pred = (w * expert_preds).sum(axis=0)  # (n_samples, n_outputs)

        if not return_std:
            return y_pred

        # Uncertainty via weighted variance of expert disagreement
        diff_sq = (expert_preds - y_pred[np.newaxis, :, :]) ** 2
        y_var = (w * diff_sq).sum(axis=0)
        y_std = np.sqrt(y_var)
        return y_pred, y_std

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "n_experts": self.n_experts,
            "expert_type": self.expert_type,
            "expert_params": self.expert_params,
            "gating_type": self.gating_type,
            "routing": self.routing,
            "max_em_iters": self.max_em_iters,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "MixtureOfExpertsSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self
