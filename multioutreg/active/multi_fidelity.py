# Copyright (c) 2026 takotime808
"""Cost-aware multi-fidelity active learning.

Each query is a pair ``(x, level)``. The acquisitions estimate how much a run
at each level would reduce uncertainty about the *highest* fidelity, divided
by its cost. See the "Multi-Fidelity Active Learning" section of
``docs/active_learning.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from multioutreg.active.acquisition import (
    _as_2d,
    _scalarize,
    aggregate,
    expected_improvement,
    output_scale,
)
from multioutreg.active.loop import ActiveLearner, heldout_metrics, space_filling


@dataclass
class FidelityInfo:
    """High-fidelity prediction at candidates, split by what each level can reduce.

    Attributes
    ----------
    mean, std : arrays, shape (n, n_outputs)
        High-fidelity predictive mean and std.
    reduction : dict[level, array (n, n_outputs)]
        Variance of the high-fidelity prediction that one run at that level
        (at the same point) would remove.
    corr : dict[level, array (n, n_outputs)]
        Correlation between that level and the high fidelity at each point.
    """

    mean: np.ndarray
    std: np.ndarray
    reduction: Dict[str, np.ndarray]
    corr: Dict[str, np.ndarray]


def _is_ar1(model) -> bool:
    return all(hasattr(model, a) for a in ("surrogate_lo_", "surrogate_delta_", "rho_"))


def _column_corr(A, B) -> np.ndarray:
    """Per-column Pearson correlation (0 where a column is constant)."""
    A, B = _as_2d(A), _as_2d(B)
    out = np.zeros(A.shape[1])
    for j in range(A.shape[1]):
        if A[:, j].std() > 0 and B[:, j].std() > 0:
            out[j] = np.corrcoef(A[:, j], B[:, j])[0, 1]
    return np.nan_to_num(out)


def _predict_level(model, X, level, levels):
    if _is_ar1(model):
        return _as_2d(model.surrogate_lo_.predict(X)) if level == levels[0] else _as_2d(model.predict(X))
    return _as_2d(model.predict(X, level=level))


def level_trust(model, levels: Sequence[str], X_hi, Y_hi, z: float = 1.0, min_points: int = 6) -> Dict[str, np.ndarray]:
    """How far the high-fidelity data supports each lower level, per output, in [0, 1].

    The squared lower confidence bound (Fisher z, ``z`` standard errors) of
    ``|corr(level prediction, Y_hi)|`` at the high-fidelity points. It is 0
    until there are ``min_points`` high-fidelity points (with 4 points, pure
    noise reaches ``|r| > 0.95`` a few percent of the time), so a lower level earns
    credit only once the data shows it predicts the high fidelity. Without it,
    a scale such as AR1's ``rho`` fitted to a handful of points can make an
    uninformative level look valuable indefinitely.
    """
    X_hi, Y_hi = np.asarray(X_hi, dtype=np.float64), _as_2d(Y_hi)
    n = len(X_hi)
    trust = {}
    for lvl in list(levels)[:-1]:
        if n < max(min_points, 4):
            trust[lvl] = np.zeros(Y_hi.shape[1])
            continue
        r = np.clip(np.abs(_column_corr(_predict_level(model, X_hi, lvl, levels), Y_hi)), 0.0, 1 - 1e-12)
        lower = np.tanh(np.arctanh(r) - z / np.sqrt(n - 3))
        trust[lvl] = np.maximum(lower, 0.0) ** 2
    return trust


def fidelity_info(model, X, levels: Sequence[str], nested: bool = True, trust: Optional[Dict[str, np.ndarray]] = None) -> FidelityInfo:
    """Per-level variance reduction for the high-fidelity prediction at ``X``.

    For AR1 models (``AdditiveCorrectionVFM``),
    ``var_hi = rho^2 var_lo + var_delta``: a low-fidelity run removes the first
    term; a high-fidelity run removes ``var_delta``, plus the first term when
    ``nested`` (the low fidelity is run at the same point too).

    For other models with ``predict(X, level=..., return_std=True)`` the
    reduction is ``corr_l^2 var_hi``, with ``corr_l`` the correlation of level
    ``l`` and high-fidelity predictions over ``X`` (Huang et al. 2006).

    ``trust`` (see :func:`level_trust`) multiplies each lower level's
    reduction, and its square root that level's correlation.
    """
    X = np.asarray(X, dtype=np.float64)
    levels = list(levels)
    top = levels[-1]
    if _is_ar1(model):
        if len(levels) != 2:
            raise ValueError("AR1 models have exactly two levels.")
        _, s_lo = model.surrogate_lo_.predict(X, return_std=True)
        _, s_d = model.surrogate_delta_.predict(X, return_std=True)
        mean, std = model.predict(X, return_std=True)
        mean, std = _as_2d(mean), _as_2d(std)
        v_lo = (model.rho_ * _as_2d(s_lo)) ** 2
        v_d = _as_2d(s_d) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            c_lo = np.where(std > 0, np.sqrt(v_lo) / std, 0.0)
        reduction = {levels[0]: v_lo, top: v_d + (v_lo if nested else 0.0)}
        corr = {levels[0]: c_lo, top: np.ones_like(std)}
    else:
        mean, std = model.predict(X, level=top, return_std=True)
        mean, std = _as_2d(mean), _as_2d(std)
        reduction, corr = {}, {}
        for lvl in levels[:-1]:
            m_l = _predict_level(model, X, lvl, levels)
            if m_l.shape != mean.shape:
                raise ValueError(f"Level {lvl!r} has {m_l.shape[1]} outputs, high fidelity has {mean.shape[1]}.")
            c = np.clip(_column_corr(m_l, mean), 0.0, 1.0)
            corr[lvl] = np.broadcast_to(c, mean.shape).copy()
            reduction[lvl] = corr[lvl] ** 2 * std ** 2
        reduction[top], corr[top] = std ** 2, np.ones_like(std)
    if trust is not None:
        for lvl, t in trust.items():
            reduction[lvl] = reduction[lvl] * t
            corr[lvl] = corr[lvl] * np.sqrt(t)
    return FidelityInfo(mean, std, reduction, corr)


def effective_costs(levels: Sequence[str], costs: Dict[str, float], nested: bool = True) -> Dict[str, float]:
    """Cost of one query per level; nested queries also pay for every lower level."""
    out, running = {}, 0.0
    for lvl in levels:
        running += costs[lvl]
        out[lvl] = running if nested else costs[lvl]
    return out


def variance_reduction_per_cost(info: FidelityInfo, eff_costs: Dict[str, float], y_scale=None, how: str = "sum", weights=None) -> np.ndarray:
    """Standardized high-fidelity variance removed per unit cost, shape (n, n_levels)."""
    cols = []
    for lvl, red in info.reduction.items():
        r = red if y_scale is None else red / np.asarray(y_scale) ** 2
        cols.append(aggregate(r, how=how, weights=weights) / eff_costs[lvl])
    return np.column_stack(cols)


def mf_expected_improvement(info: FidelityInfo, eff_costs: Dict[str, float], best: float, xi: float = 0.0, weights=None, minimize: bool = True) -> np.ndarray:
    """Multi-fidelity EI (Huang et al. 2006): ``EI_hi * corr_l * C_top / C_l``, shape (n, n_levels)."""
    ei = expected_improvement(info.mean, info.std, best, xi=xi, weights=weights, minimize=minimize)
    _, sigma = _scalarize(info.mean, info.std, weights)
    top_cost = list(eff_costs.values())[-1]
    cols = []
    for lvl, c in info.corr.items():
        # Correlation of the scalarized objective: variance-weighted mean over outputs.
        _, s_l = _scalarize(info.mean, c * info.std, weights)
        with np.errstate(divide="ignore", invalid="ignore"):
            c_g = np.where(sigma > 0, s_l / sigma, 0.0)
        cols.append(ei * c_g * top_cost / eff_costs[lvl])
    return np.column_stack(cols)


class MultiFidelityActiveLearner:
    """Choose the next ``(x, level)`` to run under a total cost budget.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning an unfitted multi-fidelity surrogate
        with ``fit({level: (X, Y)})``, e.g. ``AdditiveCorrectionVFM`` (two
        levels named ``"lo"`` / ``"hi"``) or ``StackedVFMSurrogate``.
    bounds : array-like, shape (n_features, 2)
    levels : sequence of str
        Fidelity levels, cheapest first; the last one is the target.
    costs : dict[level, float]
        Cost of one evaluation at each level.
    acquisition : {"variance", "ei"}, default "variance"
        ``"variance"``: variance reduction per cost (global accuracy).
        ``"ei"``: multi-fidelity expected improvement (optimization).
    nested : bool, default True
        A query at a level also runs every lower level at the same point.
    min_hi_fraction : float, default 0.25
        Force a high-fidelity query whenever high-fidelity spending is below
        this share of the total spent. Evidence about whether a lower level
        is informative only comes from high-fidelity runs, so this keeps it
        coming even when a lower level (spuriously) looks valuable.
    trust_z : float or None, default 1.0
        Standard errors for :func:`level_trust`; lower levels are credited
        only as far as the high-fidelity data shows they predict it. ``None``
        trusts the model's own variance split.
    """

    def __init__(
        self,
        model_factory: Callable,
        bounds,
        levels: Sequence[str],
        costs: Dict[str, float],
        acquisition: str = "variance",
        nested: bool = True,
        min_hi_fraction: float = 0.25,
        trust_z: Optional[float] = 1.0,
        how: str = "sum",
        weights=None,
        minimize: bool = True,
        xi: float = 0.0,
        n_candidates: int = 1024,
        random_state: Optional[int] = 0,
    ):
        if acquisition not in ("variance", "ei"):
            raise ValueError("acquisition must be 'variance' or 'ei'.")
        missing = set(levels) - set(costs)
        if missing:
            raise ValueError(f"No cost given for level(s) {sorted(missing)}.")
        self.model_factory = model_factory
        self.levels = list(levels)
        self.costs = {lvl: float(costs[lvl]) for lvl in self.levels}
        self.acquisition = acquisition
        self.nested = nested
        self.min_hi_fraction = min_hi_fraction
        self.trust_z = trust_z
        self.how = how
        self.weights = weights
        self.minimize = minimize
        self.xi = xi
        self.n_candidates = n_candidates
        # Reuse the single-fidelity learner for coordinates and seeding.
        self._space = ActiveLearner(lambda: None, bounds, random_state=random_state)
        self.data_: Dict[str, List[np.ndarray]] = {}
        self.spent_ = {lvl: 0.0 for lvl in self.levels}
        self.model_ = None
        self.history_: List[dict] = []

    @property
    def top(self) -> str:
        return self.levels[-1]

    @property
    def total_spent(self) -> float:
        return float(sum(self.spent_.values()))

    @property
    def eff_costs(self) -> Dict[str, float]:
        return effective_costs(self.levels, self.costs, self.nested)

    def tell(self, level: str, X, Y) -> "MultiFidelityActiveLearner":
        """Add evaluations at ``level``; charges ``costs[level]`` per row."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = _as_2d(Y)
        if level not in self.costs:
            raise ValueError(f"Unknown fidelity level {level!r}.")
        self.spent_[level] += self.costs[level] * Y.shape[0]
        if level in self.data_:
            Xo, Yo = self.data_[level]
            X, Y = np.vstack([Xo, X]), np.vstack([Yo, Y])
        self.data_[level] = [X, Y]
        self.model_ = None
        return self

    def fit(self):
        if self.model_ is None:
            self.model_ = self.model_factory().fit({k: tuple(v) for k, v in self.data_.items()})
        return self.model_

    def initial_design(self, n_per_level: Dict[str, int]) -> Dict[str, np.ndarray]:
        """Nested Sobol designs: each level's points are a prefix of the level below's."""
        n_max = max(n_per_level.values())
        X = self._space.from_unit(space_filling(n_max, self._space.n_features, "sobol", self._space._seed()))
        return {lvl: X[: n_per_level[lvl]] for lvl in self.levels}

    def score(self, X) -> np.ndarray:
        """Acquisition for every ``(x, level)``, shape (n, n_levels)."""
        model = self.fit()
        X_hi, Y_hi = self.data_[self.top]
        trust = None if self.trust_z is None else level_trust(model, self.levels, X_hi, Y_hi, self.trust_z)
        info = fidelity_info(model, X, self.levels, self.nested, trust)
        if self.acquisition == "variance":
            return variance_reduction_per_cost(info, self.eff_costs, output_scale(self.data_[self.top][1]), self.how, self.weights)
        g, _ = _scalarize(self.data_[self.top][1], np.zeros_like(self.data_[self.top][1]), self.weights)
        best = float(g.min() if self.minimize else g.max())
        return mf_expected_improvement(info, self.eff_costs, best, self.xi, self.weights, self.minimize)

    def ask(self, remaining: Optional[float] = None):
        """Next query as ``(level, x)`` with ``x`` of shape (1, n_features); None if unaffordable."""
        X = self._space.from_unit(space_filling(self.n_candidates, self._space.n_features, "sobol", self._space._seed()))
        S = self.score(X)
        eff = self.eff_costs
        affordable = np.array([remaining is None or eff[lvl] <= remaining + 1e-12 for lvl in self.levels])
        if not affordable.any():
            return None
        spent = self.total_spent
        if self.min_hi_fraction > 0 and affordable[-1] and spent > 0 and self.spent_[self.top] / spent < self.min_hi_fraction:
            affordable[:-1] = False
        S = np.where(affordable[None, :], S, -np.inf)
        i, j = np.unravel_index(int(np.argmax(S)), S.shape)
        self._last_scores = S
        return self.levels[j], X[i : i + 1]

    def _record(self, X_test, Y_test, **extra) -> dict:
        rec = {
            "cost": self.total_spent,
            **{f"n_{lvl}": len(self.data_.get(lvl, [[]])[0]) for lvl in self.levels},
            **{f"cost_{lvl}": self.spent_[lvl] for lvl in self.levels},
            **extra,
        }
        if X_test is not None:
            rec.update(heldout_metrics(self.fit(), X_test, Y_test))
        self.history_.append(rec)
        return rec

    def run(
        self,
        simulators: Dict[str, Callable[[np.ndarray], np.ndarray]],
        budget: float,
        n_init: Optional[Dict[str, int]] = None,
        X_test=None,
        Y_test=None,
        callback: Optional[Callable] = None,
    ) -> List[dict]:
        """Spend ``budget`` (including the initial design) and return :attr:`history_`.

        ``X_test`` / ``Y_test`` are high-fidelity hold-out data.
        """
        if not self.data_:
            n_init = n_init or {lvl: 2 * self._space.n_features + 2 for lvl in self.levels}
            for lvl, X0 in self.initial_design(n_init).items():
                self.tell(lvl, X0, simulators[lvl](X0))
        if self.total_spent > budget:
            raise ValueError(f"The initial design costs {self.total_spent}, over the budget of {budget}.")
        rec = self._record(X_test, Y_test, level=None)
        if callback is not None:
            callback(self, rec)
        while True:
            query = self.ask(budget - self.total_spent)
            if query is None:
                break
            level, x = query
            run_levels = self.levels[: self.levels.index(level) + 1] if self.nested else [level]
            for lvl in run_levels:
                self.tell(lvl, x, simulators[lvl](x))
            rec = self._record(X_test, Y_test, level=level, x=x.ravel())
            if callback is not None:
                callback(self, rec)
        return self.history_
