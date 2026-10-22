# Copyright (c) 2026 takotime808
"""Analytic benchmark functions for active learning, with fidelity variants.

Definitions follow the Virtual Library of Simulation Experiments
(Surjanovic & Bingham, https://www.sfu.ca/~ssurjano/). Every function takes
``X`` of shape ``(n, d)`` in its own bounds and returns ``(n, n_outputs)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from scipy.stats import qmc


def _col(y) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Raw functions
# ---------------------------------------------------------------------------


def forrester(X):
    """Forrester et al. (2008): ``(6x - 2)^2 sin(12x - 4)`` on [0, 1]."""
    x = np.asarray(X, dtype=np.float64)[:, 0]
    return _col((6 * x - 2) ** 2 * np.sin(12 * x - 4))


def forrester_lo(X):
    """Low-fidelity Forrester: ``0.5 f(x) + 10 (x - 0.5) - 5``."""
    x = np.asarray(X, dtype=np.float64)[:, :1]
    return 0.5 * forrester(X) + 10 * (x - 0.5) - 5


def branin(X):
    """Branin-Hoo on [-5, 10] x [0, 15]; global minimum 0.397887."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    b, c, t = 5.1 / (4 * np.pi ** 2), 5 / np.pi, 1 / (8 * np.pi)
    return _col((x2 - b * x1 ** 2 + c * x1 - 6) ** 2 + 10 * (1 - t) * np.cos(x1) + 10)


def currin(X):
    """Currin et al. (1988) exponential function on [0, 1]^2."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    with np.errstate(divide="ignore"):
        factor = np.where(x2 > 0, 1 - np.exp(-1 / (2 * np.where(x2 > 0, x2, 1))), 1.0)
    num = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    den = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20
    return _col(factor * num / den)


def currin_lo(X):
    """Low-fidelity Currin (Xiong et al. 2013): average of four shifted copies."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2 = X[:, 0], X[:, 1]
    h = 0.05
    x2m = np.maximum(0.0, x2 - h)

    def f(a, b):
        return currin(np.column_stack([a, b]))

    return 0.25 * (f(x1 + h, x2 + h) + f(x1 + h, x2m) + f(x1 - h, x2 + h) + f(x1 - h, x2m))


def branin_currin(X):
    """Two-output Branin (rescaled to [0, 1]^2) and Currin, as in BoTorch."""
    X = np.asarray(X, dtype=np.float64)
    U = np.column_stack([15 * X[:, 0] - 5, 15 * X[:, 1]])
    return np.hstack([branin(U), currin(X)])


def ishigami(X, a: float = 7.0, b: float = 0.1):
    """Ishigami & Homma (1990) on [-pi, pi]^3."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return _col(np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1))


_H3_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_H3_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
_H3_P = 1e-4 * np.array(
    [[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]]
)


def hartmann3(X):
    """Hartmann 3-D on [0, 1]^3; global minimum -3.86278 at (0.1146, 0.5556, 0.8525)."""
    X = np.asarray(X, dtype=np.float64)
    inner = (_H3_A[None] * (X[:, None, :] - _H3_P[None]) ** 2).sum(axis=2)
    return _col(-(np.exp(-inner) * _H3_ALPHA).sum(axis=1))


def park(X):
    """Park (1991) first function on [0, 1]^4 (x1 is kept away from 0)."""
    X = np.asarray(X, dtype=np.float64)
    x1 = np.maximum(X[:, 0], 1e-6)
    x2, x3, x4 = X[:, 1], X[:, 2], X[:, 3]
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3 ** 2) * x4 / x1 ** 2) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    return _col(term1 + term2)


def park_lo(X):
    """Low-fidelity Park (Xiong et al. 2013)."""
    X = np.asarray(X, dtype=np.float64)
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (1 + np.sin(x1)[:, None] / 10) * park(X) + _col(-2 * x1 + x2 ** 2 + x3 ** 2 + 0.5)


def _borehole(X, numerator: float, offset: float):
    X = np.asarray(X, dtype=np.float64)
    rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
    log_ratio = np.log(r / rw)
    den = log_ratio * (offset + 2 * L * Tu / (log_ratio * rw ** 2 * Kw) + Tu / Tl)
    return _col(numerator * Tu * (Hu - Hl) / den)


def borehole(X):
    """Borehole water flow (Morris et al. 1993), 8 inputs in physical units."""
    return _borehole(X, 2 * np.pi, 1.0)


def borehole_lo(X):
    """Low-fidelity borehole (Xiong et al. 2013)."""
    return _borehole(X, 5.0, 1.5)


# ---------------------------------------------------------------------------
# Benchmark container
# ---------------------------------------------------------------------------


@dataclass
class Benchmark:
    """An analytic test problem with bounds and, optionally, fidelity levels.

    ``functions`` and ``costs`` are keyed by level, ordered cheapest first;
    the last level is the high fidelity.
    """

    name: str
    bounds: np.ndarray
    functions: Dict[str, Callable[[np.ndarray], np.ndarray]]
    costs: Dict[str, float] = field(default_factory=dict)
    minimum: Optional[float] = None

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=np.float64)
        if not self.costs:
            self.costs = {lvl: 1.0 for lvl in self.functions}

    @property
    def levels(self):
        return list(self.functions)

    @property
    def high(self) -> str:
        return self.levels[-1]

    @property
    def n_features(self) -> int:
        return self.bounds.shape[0]

    def __call__(self, X, level: Optional[str] = None) -> np.ndarray:
        return self.functions[level or self.high](np.asarray(X, dtype=np.float64))

    def sample(self, n: int, method: str = "sobol", seed=None) -> np.ndarray:
        """Space-filling (``"sobol"``, ``"lhs"``) or ``"random"`` points in bounds."""
        d = self.n_features
        if method == "sobol":
            U = qmc.Sobol(d, scramble=True, seed=seed).random(n)
        elif method == "lhs":
            U = qmc.LatinHypercube(d, seed=seed).random(n)
        elif method == "random":
            U = np.random.default_rng(seed).random((n, d))
        else:
            raise ValueError(f"Unknown sampling method {method!r}.")
        return qmc.scale(U, self.bounds[:, 0], self.bounds[:, 1])


_BOREHOLE_BOUNDS = [
    [0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110],
    [63.1, 116], [700, 820], [1120, 1680], [9855, 12045],
]


def get_benchmark(name: str, cost_ratio: float = 10.0) -> Benchmark:
    """Return a :class:`Benchmark` by name.

    Multi-fidelity benchmarks have levels ``"lo"`` and ``"hi"`` with costs
    ``1`` and ``cost_ratio``.
    """
    mf = {"lo": 1.0, "hi": float(cost_ratio)}
    table = {
        "forrester": lambda: Benchmark("forrester", [[0, 1]], {"lo": forrester_lo, "hi": forrester}, dict(mf), -6.02074),
        "branin": lambda: Benchmark("branin", [[-5, 10], [0, 15]], {"hi": branin}, minimum=0.397887),
        "currin": lambda: Benchmark("currin", [[0, 1], [0, 1]], {"lo": currin_lo, "hi": currin}, dict(mf)),
        "branin_currin": lambda: Benchmark("branin_currin", [[0, 1], [0, 1]], {"hi": branin_currin}),
        "ishigami": lambda: Benchmark("ishigami", [[-np.pi, np.pi]] * 3, {"hi": ishigami}),
        "hartmann3": lambda: Benchmark("hartmann3", [[0, 1]] * 3, {"hi": hartmann3}, minimum=-3.86278),
        "park": lambda: Benchmark("park", [[0, 1]] * 4, {"lo": park_lo, "hi": park}, dict(mf)),
        "borehole": lambda: Benchmark("borehole", _BOREHOLE_BOUNDS, {"lo": borehole_lo, "hi": borehole}, dict(mf)),
    }
    if name not in table:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(table)}.")
    return table[name]()


BENCHMARKS = (
    "forrester", "branin", "currin", "branin_currin",
    "ishigami", "hartmann3", "park", "borehole",
)
