# Copyright (c) 2026 takotime808
"""Price <-> change transforms used by ``ts-pipeline --transform``."""

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.returns import to_changes, to_levels


def _prices(n=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.mark.parametrize("kind", ["diff", "log-return"])
def test_changes_cumulate_back_to_the_levels(kind):
    p = _prices()
    changes = to_changes(p, kind)
    assert len(changes) == len(p) - 1 and changes.index[0] == p.index[1]
    # Forecasting the "future" p[11:] from p[10] with the true changes recovers it.
    np.testing.assert_allclose(to_levels(p.iloc[10], changes.iloc[10:], kind), p.iloc[11:])


def test_log_returns_values():
    p = pd.Series([100.0, 110.0, 99.0])
    np.testing.assert_allclose(to_changes(p, "log-return"), np.log([1.1, 0.9]))
    np.testing.assert_allclose(to_levels(99.0, [0.0, np.log(2)], "log-return"), [99, 198])


def test_none_is_the_identity():
    p = _prices()
    assert to_changes(p, "none") is p
    np.testing.assert_allclose(to_levels(5.0, [1.0, 2.0], "none"), [1.0, 2.0])


def test_errors():
    with pytest.raises(ValueError, match="strictly positive"):
        to_changes(pd.Series([1.0, 0.0, 2.0]), "log-return")
    with pytest.raises(ValueError, match="transform must be one of"):
        to_changes(_prices(), "pct")
    with pytest.raises(ValueError, match="transform must be one of"):
        to_levels(1.0, [0.1], "pct")
