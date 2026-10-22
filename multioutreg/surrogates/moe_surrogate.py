# Copyright (c) 2026 takotime808

"""Mixture of Experts surrogate model."""

from __future__ import annotations

from typing import Literal

import numpy as np

from multioutreg.surrogates.conformal_mixin import ConformalMixin


class MixtureOfExpertsSurrogate(ConformalMixin):
    """Mixture of Experts surrogate with learned gating network.

    Specializes K expert regressors to different regions of input space
    using a gating network trained via a hard-EM loop. Each E-step reassigns
    every sample to the expert with the highest gating prior times Gaussian
    likelihood of that expert's residual, so assignments follow expert fit.

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

    @staticmethod
    def _expert_outputs(expert, X: np.ndarray, return_std: bool):
        """Expert prediction (and std, 0 if unsupported) as (n_samples, n_outputs)."""
        if return_std:
            try:
                pred, std = expert.predict(X, return_std=True)
            except TypeError:
                pred = expert.predict(X)
                std = np.zeros_like(np.asarray(pred, dtype=np.float64))
            pred, std = np.asarray(pred), np.asarray(std)
            return pred.reshape(len(X), -1), std.reshape(len(X), -1)
        return np.asarray(expert.predict(X)).reshape(len(X), -1)

    def _reassign(self, X, Y, labels, experts, gating) -> np.ndarray:
        """E-step: argmax_k log pi_k(x) + log N(y | mu_k(x), sigma^2) per sample.

        sigma^2 is a per-output noise variance pooled over every sample's
        current expert, so outputs on different scales are weighted fairly.
        """
        n = X.shape[0]
        preds = np.stack([self._expert_outputs(e, X, False) for e in experts])
        resid2 = (Y[np.newaxis] - preds) ** 2               # (K, n, n_outputs)
        sigma2 = resid2[labels, np.arange(n)].mean(axis=0)
        sigma2 = np.maximum(sigma2, 1e-12 * np.maximum(Y.var(axis=0), 1.0))
        loglik = -0.5 * (resid2 / sigma2 + np.log(2 * np.pi * sigma2)).sum(axis=2)
        log_prior = np.log(np.clip(gating.predict_proba(X), 1e-12, None))
        score = log_prior + loglik.T                        # (n, K)
        new_labels = score.argmax(axis=1)
        # An expert that lost all its samples would be refit on everything and
        # never recover; give it the worst-fit samples instead.
        for k in range(self.n_experts):
            if not np.any(new_labels == k):
                fit = score[np.arange(n), new_labels]
                worst = np.argsort(fit)[: max(1, n // (2 * self.n_experts))]
                new_labels[worst] = k
        return new_labels

    def _initial_labels(self, X, Y, rng) -> np.ndarray:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        if X.shape[0] < self.n_experts:
            return rng.integers(0, self.n_experts, size=X.shape[0])
        Z = StandardScaler().fit_transform(np.hstack([X, Y]))
        seed = int(rng.integers(0, 2**31 - 1))
        return KMeans(self.n_experts, n_init=3, random_state=seed).fit_predict(Z)

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

        # Initialise by k-means on standardised (X, Y): random labels give
        # near-identical experts, and one then takes every sample.
        labels = self._initial_labels(X, Y, rng)
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

            # E-step: reassign each sample by gating prior x expert likelihood
            new_labels = self._reassign(X, Y, labels, experts, gating)

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
            If True, also return the mixture's predictive std. Soft routing uses
            the law of total variance, ``sum_k w_k (sigma_k^2 + (mu_k - mu)^2)``;
            hard routing returns the chosen expert's own std. Experts without
            native uncertainty contribute sigma_k = 0.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std : np.ndarray, shape (n_samples, n_outputs)  [only if return_std=True]
        """
        if not hasattr(self, "experts_"):
            raise AttributeError("MixtureOfExpertsSurrogate is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        weights = self.get_routing_weights(X)  # (n_samples, n_experts)

        outputs = [self._expert_outputs(e, X, return_std) for e in self.experts_]
        # Stack to (n_experts, n_samples, n_outputs)
        if return_std:
            expert_preds = np.stack([o[0] for o in outputs], axis=0)
            expert_stds = np.stack([o[1] for o in outputs], axis=0)
        else:
            expert_preds = np.stack(outputs, axis=0)

        if self.routing == "hard":
            assignment = weights.argmax(axis=1)  # (n_samples,)
            rows = np.arange(X.shape[0])
            y_pred = expert_preds[assignment, rows, :]
            if return_std:
                return y_pred, expert_stds[assignment, rows, :]
            return y_pred

        # Soft routing: weighted sum over experts
        # w shape: (n_experts, n_samples, 1)
        w = weights.T[:, :, np.newaxis]
        y_pred = (w * expert_preds).sum(axis=0)  # (n_samples, n_outputs)

        if not return_std:
            return y_pred

        # Law of total variance: within-expert + between-expert (disagreement)
        diff_sq = (expert_preds - y_pred[np.newaxis, :, :]) ** 2
        y_var = (w * (expert_stds ** 2 + diff_sq)).sum(axis=0)
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
