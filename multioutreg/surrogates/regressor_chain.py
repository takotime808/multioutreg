# Copyright (c) 2026 takotime808

"""Regressor chain with Monte Carlo propagation of upstream uncertainty.

A regressor chain fits the outputs in sequence: output ``order[k]`` is modelled
from the inputs plus outputs ``order[:k]``, so later outputs can depend on
earlier ones (a stress that follows from a computed displacement, say). Like
:class:`sklearn.multioutput.RegressorChain`, each link is trained on the true
upstream outputs.

At prediction time the upstream outputs are unknown. Feeding their predicted
means forward (the plug-in chain) gives the mean but drops their uncertainty.
Here the std and covariance come from sampling instead: draw each upstream
output from ``N(mean, std^2)`` of its link, feed the draws forward, and combine
the downstream link's own variance with the spread of its means across draws
(law of total variance)::

    var_j(x) = E_s[std_j(x, y_s)^2] + Var_s[mean_j(x, y_s)]

The first link has no upstream outputs, so its std is exact. The returned mean
is the plug-in chain, so ``predict(X)`` and ``predict(X, return_std=True)``
agree; for linear links it equals the propagated mean.
"""

import numpy as np

from multioutreg.surrogates.conformal_mixin import ConformalMixin

# Rows per link.predict call during sampling, to bound kernel-matrix memory.
_CHUNK_ROWS = 20_000


def _link_predict(link, X, return_std):
    """Mean (and std) of a one-output link, as 1-D arrays."""
    means, stds = [], []
    for start in range(0, len(X), _CHUNK_ROWS):
        chunk = X[start:start + _CHUNK_ROWS]
        if not return_std:
            means.append(np.asarray(link.predict(chunk)).ravel())
            continue
        out = link.predict(chunk, return_std=True)
        if isinstance(out, tuple) and len(out) == 2:
            mu, sd = out
        else:
            mu, sd = link.predict(chunk), np.zeros(len(chunk))
        means.append(np.asarray(mu).ravel())
        stds.append(np.asarray(sd).ravel())
    if not return_std:
        return np.concatenate(means)
    return np.concatenate(means), np.concatenate(stds)


class RegressorChainSurrogate(ConformalMixin):
    """Chain of one-output surrogates with sampled uncertainty propagation.

    Parameters
    ----------
    surrogate_cls : type | None
        Surrogate used for every link; it must accept ``fit(X, y)`` and
        ``predict(X, return_std=True)``. Defaults to
        ``GaussianProcessSurrogate``.
    surrogate_kwargs : dict | None
        Passed to ``surrogate_cls``.
    order : sequence of int | None
        Order in which outputs are chained (a permutation of the output
        indices). Defaults to the column order.
    n_samples : int
        Monte Carlo draws used for the std and covariance.
    random_state : int | None
        Seed for the draws, so repeated calls give the same std.

    Attributes
    ----------
    order_ : np.ndarray
    links_ : list
        Fitted surrogate per position in the chain.
    """

    _multi_output = True  # signals AutoDetect to evaluate on full Y

    def __init__(self, surrogate_cls=None, surrogate_kwargs=None, order=None,
                 n_samples=100, random_state=0):
        self.surrogate_cls = surrogate_cls
        self.surrogate_kwargs = surrogate_kwargs
        self.order = order
        self.n_samples = n_samples
        self.random_state = random_state

    def _make_link(self):
        cls = self.surrogate_cls
        if cls is None:
            from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

            cls = GaussianProcessSurrogate
        return cls(**(self.surrogate_kwargs or {}))

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        m = Y.shape[1]
        order = np.arange(m) if self.order is None else np.asarray(self.order, dtype=int)
        if sorted(order.tolist()) != list(range(m)):
            raise ValueError(f"order must be a permutation of range({m}), got {self.order!r}")
        self.order_ = order
        self.n_outputs_ = m
        self.links_ = []
        for k, j in enumerate(order):
            X_aug = np.hstack([X, Y[:, order[:k]]])
            self.links_.append(self._make_link().fit(X_aug, Y[:, [j]]))
        return self

    def _plugin_mean(self, X):
        mean = np.empty((len(X), self.n_outputs_))
        for k, (j, link) in enumerate(zip(self.order_, self.links_)):
            X_aug = np.hstack([X, mean[:, self.order_[:k]]])
            mean[:, j] = _link_predict(link, X_aug, return_std=False)
        return mean

    def sample(self, X, n_samples=None, random_state=None):
        """Joint draws of the outputs, shape ``(n_samples, n, n_outputs)``."""
        X = np.asarray(X, dtype=float)
        draws, _, _ = self._propagate(X, n_samples, random_state)
        return draws

    def _propagate(self, X, n_samples=None, random_state=None):
        S = self.n_samples if n_samples is None else n_samples
        if S < 2:
            raise ValueError(f"n_samples must be >= 2, got {S}")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        n = len(X)
        X_rep = np.tile(X, (S, 1))
        draws = np.empty((S * n, self.n_outputs_))
        link_mean = np.empty_like(draws)
        link_var = np.empty_like(draws)
        for k, (j, link) in enumerate(zip(self.order_, self.links_)):
            if k == 0:
                mu, sd = _link_predict(link, X, return_std=True)
                mu, sd = np.tile(mu, S), np.tile(sd, S)
            else:
                X_aug = np.hstack([X_rep, draws[:, self.order_[:k]]])
                mu, sd = _link_predict(link, X_aug, return_std=True)
            link_mean[:, j], link_var[:, j] = mu, np.square(sd)
            draws[:, j] = mu + sd * rng.standard_normal(S * n)
        shape = (S, n, self.n_outputs_)
        return draws.reshape(shape), link_mean.reshape(shape), link_var.reshape(shape)

    def predict(self, X, return_std=False, return_cov=False):
        """Predict outputs, optionally with per-output std or the full covariance.

        ``return_cov=True`` returns ``(mean, cov)`` with ``cov`` of shape
        ``(n_samples, n_outputs, n_outputs)``, estimated from the joint draws,
        and takes precedence over ``return_std``.
        """
        X = np.asarray(X, dtype=float)
        mean = self._plugin_mean(X)
        if not (return_std or return_cov):
            return mean
        draws, link_mean, link_var = self._propagate(X)
        if return_cov:
            centred = draws - draws.mean(axis=0)
            cov = np.einsum("sni,snj->nij", centred, centred) / (len(draws) - 1)
            return mean, cov
        var = link_var.mean(axis=0) + link_mean.var(axis=0)
        return mean, np.sqrt(var)
