# Copyright (c) 2026 takotime808
"""Single-fidelity active-learning loop with an ask / tell interface."""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from scipy.stats import qmc

from multioutreg.active.acquisition import (
    ACQUISITIONS,
    _as_2d,
    _scalarize,
    output_scale,
    predict_mean_std,
)
from multioutreg.active.batch import select_batch


def space_filling(n: int, n_features: int, method: str = "sobol", seed=None) -> np.ndarray:
    """``n`` points in the unit cube from a scrambled Sobol, LHS or random design."""
    if method == "sobol":
        sampler = qmc.Sobol(n_features, scramble=True, seed=seed)
        return sampler.random_base2(max(0, math.ceil(math.log2(max(n, 1)))))[:n]
    if method == "lhs":
        return qmc.LatinHypercube(n_features, seed=seed).random(n)
    if method == "random":
        return np.random.default_rng(seed).random((n, n_features))
    raise ValueError(f"Unknown design method {method!r}.")


def heldout_metrics(model, X_test, Y_test) -> Dict[str, float]:
    """RMSE and range-free normalized RMSE (RMSE / std of Y_test), averaged over outputs."""
    Y_test = _as_2d(Y_test)
    pred = _as_2d(model.predict(X_test))
    rmse = np.sqrt(((pred - Y_test) ** 2).mean(axis=0))
    return {"rmse": float(rmse.mean()), "nrmse": float((rmse / output_scale(Y_test)).mean())}


class ActiveLearner:
    """Sequential design for a multi-output surrogate on a box-shaped domain.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning an unfitted surrogate with
        ``fit(X, Y)`` and ``predict(X, return_std=True)``. The model is refit
        from scratch after every :meth:`tell`.
    bounds : array-like, shape (n_features, 2)
    acquisition : str or callable, default "max_variance"
        One of ``"max_variance"``, ``"ucb"``, ``"ei"``, ``"pi"``, ``"random"``,
        or ``f(mean, std, **kwargs) -> scores``.
    acquisition_kwargs : dict, optional
        Extra arguments for the acquisition (``beta``, ``xi``, ``how``).
    weights : sequence of float, optional
        Per-output weights: aggregation weights for ``max_variance``,
        scalarization weights for ``ucb`` / ``ei`` / ``pi``.
    minimize : bool, default True
        Direction for the optimization acquisitions.
    batch_size : int, default 1
    n_candidates : int, default 1024
        Size of the fresh Sobol candidate pool drawn on every :meth:`ask`.
    length_scale, penalize_train
        Passed to :func:`multioutreg.active.batch.select_batch`.
    allow_zero_std : bool, default False
        See :func:`multioutreg.active.acquisition.predict_mean_std`.
    random_state : int or None, default 0
    """

    def __init__(
        self,
        model_factory: Callable,
        bounds,
        acquisition: Union[str, Callable] = "max_variance",
        acquisition_kwargs: Optional[dict] = None,
        weights=None,
        minimize: bool = True,
        batch_size: int = 1,
        n_candidates: int = 1024,
        length_scale: Optional[float] = None,
        penalize_train: bool = False,
        allow_zero_std: bool = False,
        random_state: Optional[int] = 0,
    ):
        self.model_factory = model_factory
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_features, 2).")
        if isinstance(acquisition, str) and acquisition not in {*ACQUISITIONS, "random"}:
            raise ValueError(f"Unknown acquisition {acquisition!r}.")
        self.acquisition = acquisition
        self.acquisition_kwargs = dict(acquisition_kwargs or {})
        self.weights = weights
        self.minimize = minimize
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.length_scale = length_scale
        self.penalize_train = penalize_train
        self.allow_zero_std = allow_zero_std
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.X_ = np.empty((0, self.n_features))
        self.Y_: Optional[np.ndarray] = None
        self.model_ = None
        self.history_: List[dict] = []

    # -- coordinates -------------------------------------------------------

    @property
    def n_features(self) -> int:
        return self.bounds.shape[0]

    def to_unit(self, X) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(X, dtype=np.float64) - lo) / (hi - lo)

    def from_unit(self, U) -> np.ndarray:
        return qmc.scale(np.asarray(U, dtype=np.float64), self.bounds[:, 0], self.bounds[:, 1])

    def _seed(self) -> int:
        return int(self._rng.integers(2 ** 31))

    # -- data --------------------------------------------------------------

    def initial_design(self, n: int, method: str = "sobol") -> np.ndarray:
        """Space-filling starting points in the domain (not yet evaluated)."""
        return self.from_unit(space_filling(n, self.n_features, method, self._seed()))

    def tell(self, X, Y) -> "ActiveLearner":
        """Add evaluated points; the model is refit on the next :meth:`fit`/:meth:`ask`."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = _as_2d(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}.")
        self.X_ = np.vstack([self.X_, X])
        self.Y_ = Y if self.Y_ is None else np.vstack([self.Y_, Y])
        self.model_ = None
        return self

    def fit(self):
        """Fit a fresh model on all data told so far and return it."""
        if self.Y_ is None or len(self.X_) < 2:
            raise RuntimeError("tell() at least 2 evaluated points before fitting.")
        if self.model_ is None:
            self.model_ = self.model_factory().fit(self.X_, self.Y_)
        return self.model_

    # -- acquisition -------------------------------------------------------

    def best_observed(self) -> float:
        g, _ = _scalarize(self.Y_, np.zeros_like(self.Y_), self.weights)
        return float(g.min() if self.minimize else g.max())

    def score(self, X) -> np.ndarray:
        """Acquisition value at ``X`` (larger is better)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.acquisition == "random":
            return self._rng.random(X.shape[0])
        model = self.fit()
        mean, std = predict_mean_std(model, X, allow_zero_std=self.allow_zero_std)
        kw = dict(self.acquisition_kwargs)
        name = self.acquisition if isinstance(self.acquisition, str) else None
        if name == "max_variance":
            kw.setdefault("y_scale", output_scale(self.Y_))
            kw.setdefault("weights", self.weights)
        elif name in ("ucb", "ei", "pi"):
            kw.setdefault("weights", self.weights)
            kw.setdefault("minimize", self.minimize)
            if name != "ucb":
                kw.setdefault("best", self.best_observed())
        fn = ACQUISITIONS[name] if name else self.acquisition
        return np.asarray(fn(mean, std, **kw), dtype=np.float64).ravel()

    def ask(self, n: Optional[int] = None) -> np.ndarray:
        """Return the next ``n`` (default ``batch_size``) points to evaluate."""
        n = n or self.batch_size
        U = space_filling(max(self.n_candidates, n), self.n_features, "sobol", self._seed())
        scores = self.score(self.from_unit(U))
        self._last_scores = scores
        idx = select_batch(
            U, scores, n,
            U_train=self.to_unit(self.X_),
            length_scale=self.length_scale,
            penalize_train=self.penalize_train,
        )
        return self.from_unit(U[idx])

    # -- loop --------------------------------------------------------------

    def _record(self, X_test, Y_test, **extra) -> dict:
        rec = {"n_evals": len(self.X_), "best": self.best_observed(), **extra}
        if X_test is not None:
            rec.update(heldout_metrics(self.fit(), X_test, Y_test))
        self.history_.append(rec)
        return rec

    def run(
        self,
        simulator: Callable[[np.ndarray], np.ndarray],
        n_init: int = 10,
        n_iter: int = 10,
        init_method: str = "sobol",
        X_test=None,
        Y_test=None,
        max_evals: Optional[int] = None,
        rtol: Optional[float] = None,
        patience: int = 2,
        callback: Optional[Callable[["ActiveLearner", dict], None]] = None,
    ) -> List[dict]:
        """Run the loop with a Python ``simulator(X) -> Y``.

        Stops after ``n_iter`` batches, when ``max_evals`` evaluations are
        reached, or (with ``rtol`` and test data) when held-out NRMSE has not
        improved by a relative ``rtol`` for ``patience`` iterations.

        Returns
        -------
        list of dict
            :attr:`history_`, one record per fitted model.
        """
        if self.Y_ is None:
            X0 = self.initial_design(n_init, init_method)
            self.tell(X0, simulator(X0))
        stale = 0
        for it in range(n_iter + 1):
            rec = self._record(X_test, Y_test, iteration=it)
            if callback is not None:
                callback(self, rec)
            if it == n_iter:
                break
            if rtol is not None and "nrmse" in rec and len(self.history_) > 1:
                prev = min(h["nrmse"] for h in self.history_[:-1])
                stale = stale + 1 if rec["nrmse"] > prev * (1 - rtol) else 0
                if stale >= patience:
                    break
            n = self.batch_size
            if max_evals is not None:
                n = min(n, max_evals - len(self.X_))
                if n <= 0:
                    break
            X_new = self.ask(n)
            self.tell(X_new, simulator(X_new))
        return self.history_
