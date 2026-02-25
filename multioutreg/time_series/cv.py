# Copyright (c) 2025 takotime808

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from multioutreg.time_series.metrics import smape, mase, weighted_quantile_loss
from multioutreg.time_series.chronos_adapter import ForecastResult


@dataclass
class TSFoldResult:
    """Result for a single walk-forward fold."""
    fold_idx: int
    train_size: int
    test_size: int
    y_true: np.ndarray          # shape (horizon,)
    y_pred: np.ndarray          # shape (horizon,)
    quantiles: Optional[np.ndarray]    # shape (n_quantiles, horizon) or None
    q_levels: Optional[Tuple[float, ...]]
    smape: float
    mase: float
    wql: Optional[float]        # None if no quantiles


def walk_forward_splits(
    n: int,
    min_train: int,
    horizon: int,
    step: int = 1,
    max_train: Optional[int] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward (train_idx, test_idx) pairs.

    Parameters
    ----------
    n : int
        Total series length.
    min_train : int
        Minimum number of training observations.
    horizon : int
        Forecast horizon (test window size).
    step : int, default 1
        Number of steps to advance the window between folds.
    max_train : int | None
        If given, applies a rolling window (training set size capped at max_train).
        None => expanding window (training set always grows).

    Yields
    ------
    train_idx : np.ndarray of int
    test_idx  : np.ndarray of int
    """
    t = min_train
    while t + horizon <= n:
        if max_train is not None:
            start = max(0, t - max_train)
        else:
            start = 0
        train_idx = np.arange(start, t)
        test_idx = np.arange(t, min(t + horizon, n))
        yield train_idx, test_idx
        t += step


class WalkForwardCV:
    """Walk-forward cross-validator for time series forecasters.

    Fits the forecaster on each training fold and evaluates on the test fold,
    computing SMAPE, MASE, and WQL (when quantile forecasts are available).

    The forecaster must implement::

        forecaster.fit(train_series)   # 1D array or pd.Series
        forecaster.predict(horizon, quantiles=quantiles)  # -> ForecastResult

    Parameters
    ----------
    min_train : int, default 30
        Minimum training observations.
    horizon : int, default 1
        Steps to forecast per fold.
    step : int, default 1
        Steps to advance the training window between folds.
    max_train : int | None, default None
        Expanding (None) or rolling window.
    seasonality : int, default 1
        Seasonal period used in the MASE denominator.

    Examples
    --------
    >>> cv = WalkForwardCV(min_train=50, horizon=5, step=5)
    >>> results = cv.evaluate(series, forecaster)
    >>> print(cv.summary(results))
    """

    def __init__(
        self,
        min_train: int = 30,
        horizon: int = 1,
        step: int = 1,
        max_train: Optional[int] = None,
        seasonality: int = 1,
    ):
        self.min_train = min_train
        self.horizon = horizon
        self.step = step
        self.max_train = max_train
        self.seasonality = seasonality

    def split(
        self,
        series: np.ndarray | pd.Series,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) pairs without fitting any model.

        Parameters
        ----------
        series : 1D time series

        Yields
        ------
        train_idx, test_idx : np.ndarray
        """
        arr = _to_array(series)
        return walk_forward_splits(
            n=len(arr),
            min_train=self.min_train,
            horizon=self.horizon,
            step=self.step,
            max_train=self.max_train,
        )

    def evaluate(
        self,
        series: np.ndarray | pd.Series,
        forecaster: Any,
        quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    ) -> list[TSFoldResult]:
        """Fit forecaster on each training fold and evaluate on the test fold.

        Parameters
        ----------
        series : 1D time series
        forecaster : Any
            Must have ``fit(train_series)`` and
            ``predict(horizon, quantiles=...) -> ForecastResult``.
        quantiles : sequence of float

        Returns
        -------
        list[TSFoldResult]
        """
        arr = _to_array(series)
        results = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            walk_forward_splits(
                n=len(arr),
                min_train=self.min_train,
                horizon=self.horizon,
                step=self.step,
                max_train=self.max_train,
            )
        ):
            train = arr[train_idx]
            test = arr[test_idx]
            h = len(test_idx)

            forecaster.fit(train)
            result: ForecastResult = forecaster.predict(horizon=h, quantiles=quantiles)

            # Extract median as point prediction
            q_tup = tuple(float(q) for q in quantiles)
            q_arr = result.quantiles[0]  # shape (Q, H)

            if 0.5 in q_tup:
                med_idx = q_tup.index(0.5)
                y_pred = q_arr[med_idx]
            else:
                y_pred = q_arr.mean(axis=0)

            s = smape(test, y_pred)
            m = mase(test, y_pred, seasonality=self.seasonality)

            wql_val: Optional[float] = None
            if q_arr is not None:
                wql_val = float(weighted_quantile_loss(test, q_arr, list(q_tup)))

            results.append(TSFoldResult(
                fold_idx=fold_idx,
                train_size=len(train),
                test_size=len(test),
                y_true=test,
                y_pred=y_pred,
                quantiles=q_arr,
                q_levels=q_tup,
                smape=float(s),
                mase=float(m),
                wql=wql_val,
            ))

        return results

    def summary(self, fold_results: list[TSFoldResult]) -> dict:
        """Aggregate fold results into mean ± std of each metric.

        Returns
        -------
        dict with keys: mean_smape, std_smape, mean_mase, std_mase,
                        mean_wql, std_wql, n_folds
        """
        smapes = np.array([r.smape for r in fold_results])
        mases = np.array([r.mase for r in fold_results])
        wqls = np.array([r.wql for r in fold_results if r.wql is not None])

        return {
            "mean_smape": float(smapes.mean()),
            "std_smape": float(smapes.std()),
            "mean_mase": float(mases.mean()),
            "std_mase": float(mases.std()),
            "mean_wql": float(wqls.mean()) if len(wqls) else None,
            "std_wql": float(wqls.std()) if len(wqls) else None,
            "n_folds": len(fold_results),
        }


class TimeSeriesSplitWrapper:
    """sklearn-compatible cross-validator wrapping walk_forward_splits.

    Allows using ``WalkForwardCV`` with sklearn utilities (e.g., ``cross_val_score``).

    Parameters
    ----------
    n_splits : int, default 5
        Approximate number of folds (used to compute ``step``).
    min_train : int, default 30
    horizon : int, default 1
    """

    def __init__(self, n_splits: int = 5, min_train: int = 30, horizon: int = 1):
        self.n_splits = n_splits
        self.min_train = min_train
        self.horizon = horizon

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Any = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) arrays compatible with sklearn."""
        n = len(X)
        available = n - self.min_train - self.horizon
        step = max(1, available // self.n_splits)
        return walk_forward_splits(
            n=n,
            min_train=self.min_train,
            horizon=self.horizon,
            step=step,
        )


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _to_array(series: np.ndarray | pd.Series) -> np.ndarray:
    if isinstance(series, pd.Series):
        return series.to_numpy(dtype=float)
    return np.asarray(series, dtype=float)
