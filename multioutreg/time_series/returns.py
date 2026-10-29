# Copyright (c) 2026 takotime808
"""Price <-> change transforms for forecasting financial series.

Prices are close to a random walk: non-stationary, with a scale that drifts.
Models are fitted to the changes instead -- simple differences or log returns
-- and their forecast is cumulated back onto the last observed level:

    log returns  r_t = log(p_t / p_{t-1})   ->   p_{T+k} = p_T * exp(r_{T+1} + ... + r_{T+k})
    differences  d_t = p_t - p_{t-1}        ->   p_{T+k} = p_T + d_{T+1} + ... + d_{T+k}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Transforms accepted by :func:`to_changes` / :func:`to_levels`.
TRANSFORMS = ("none", "diff", "log-return")


def _check(kind: str) -> None:
    if kind not in TRANSFORMS:
        raise ValueError(f"transform must be one of {TRANSFORMS}, got {kind!r}")


def to_changes(levels: pd.Series, kind: str) -> pd.Series:
    """Period-to-period changes of *levels*; the first period is dropped.

    ``kind='none'`` returns *levels* unchanged. Log returns need every level
    to be positive.
    """
    _check(kind)
    if kind == "none":
        return levels
    if kind == "log-return":
        if (levels <= 0).any():
            raise ValueError("log returns need strictly positive values; use --transform diff")
        return np.log(levels).diff().iloc[1:]
    return levels.diff().iloc[1:]


def to_levels(last_level: float, changes, kind: str) -> np.ndarray:
    """Cumulate forecast *changes* onto *last_level*: the forecast as levels."""
    _check(kind)
    changes = np.asarray(changes, dtype=float)
    if kind == "none":
        return changes
    if kind == "log-return":
        return float(last_level) * np.exp(np.cumsum(changes))
    return float(last_level) + np.cumsum(changes)
