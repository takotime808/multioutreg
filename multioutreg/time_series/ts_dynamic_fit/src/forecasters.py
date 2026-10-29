# Copyright (c) 2026 takotime808
"""ARIMA / SARIMA / LSTM / naive forecasters with one interface (audit item 6).

Every forecaster has ``fit(y_train)`` and ``forecast(horizon) -> np.ndarray``,
and the forecast is always on the **original scale** of ``y_train``, so all
models can be scored against the same held-out values by one metric function
(:mod:`multioutreg.time_series.ts_dynamic_fit.src.evaluate`).

ARIMA / SARIMA choose ``d`` with the ADF test (at most :data:`MAX_D`
differences) and pass it to statsmodels, which integrates the forecast back to
levels. ``fit(y, order=...)`` skips the order search and only re-estimates the
parameters; rolling-origin evaluation uses that to search the order once, on
the first (smallest) training window.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Optional

import numpy as np

#: Largest number of differences the ADF loop may apply (audit item 24).
MAX_D = 2
#: Largest AR / MA order in the ARIMA grid (p, q in 0..7).
ARIMA_MAX_ORDER = 7


def _as_1d(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("the training series must be non-empty and finite")
    return y


def _standardize(y):
    """``(y - mean) / sd`` and the (mean, sd) to undo it.

    ARIMA is equivariant under affine maps of y, but statsmodels' optimiser is
    not: on a series around 1e6 it can converge elsewhere than on the same
    series around 1e3. Fitting on the standardised series makes the order
    search and forecasts scale-invariant.
    """
    mu, sd = float(np.mean(y)), float(np.std(y))
    sd = sd if sd > 0 else 1.0
    return (y - mu) / sd, (mu, sd)


def choose_d(y, alpha: float = 0.05, max_d: int = MAX_D) -> int:
    """Number of differences until the ADF test rejects a unit root (at most ``max_d``)."""
    from statsmodels.tsa.stattools import adfuller

    d, z = 0, np.asarray(y, dtype=float)
    while d < max_d:
        if np.ptp(z) == 0:          # constant: stationary, and adfuller would fail
            break
        if adfuller(z)[1] <= alpha:
            break
        z = np.diff(z)
        d += 1
    return d


class NaiveForecaster:
    """Last value, or the value one season back when ``seasonality >= 2``."""

    name = "Naive"

    def __init__(self, seasonality: int = 1):
        self.seasonality = seasonality

    def fit(self, y_train, order=None) -> "NaiveForecaster":
        self.y_ = _as_1d(y_train)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        m = self.seasonality if self.seasonality >= 2 and self.y_.size >= self.seasonality else 1
        last = self.y_[-m:]
        return np.resize(last, horizon).astype(float)


class ArimaForecaster:
    """ARIMA(p, d, q): ``d`` from the ADF test, (p, q) by AIC over 0..7."""

    name = "ARIMA"

    def __init__(self, max_order: int = ARIMA_MAX_ORDER, n_jobs: int = 1):
        self.max_order = max_order
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_one(y, order):
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return sm.tsa.ARIMA(y, order=order).fit()
            except Exception:
                return None

    def fit(self, y_train, order=None) -> "ArimaForecaster":
        y, self.scale_ = _standardize(_as_1d(y_train))
        if order is None:
            d = choose_d(y)
            grid = [(p, d, q) for p, q in itertools.product(
                range(self.max_order + 1), repeat=2)]
            fits = _parallel(self._fit_one, [(y, o) for o in grid], self.n_jobs)
            scored = [(r.aic, o, r) for o, r in zip(grid, fits)
                      if r is not None and np.isfinite(r.aic)]
            if not scored:
                raise RuntimeError("ARIMA: no (p, d, q) order could be fitted")
            _, order, self.results_ = min(scored, key=lambda t: t[0])
        else:
            self.results_ = self._fit_one(y, tuple(order))
            if self.results_ is None:
                raise RuntimeError(f"ARIMA{tuple(order)} could not be fitted")
        self.order_ = tuple(order)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        mu, sd = self.scale_
        return mu + sd * np.asarray(self.results_.forecast(horizon), dtype=float)


class SarimaForecaster:
    """SARIMA(p, d, q)(P, 0, Q, s) over p, q, P, Q in {0, 1}.

    ``d`` comes from the ADF test and ``s`` from the ACF of the differenced
    series (or ``seasonality``). Without a season (s < 2, or fewer than two
    seasons plus two points of data) it reduces to ARIMA(p, d, q), p, q <= 1.
    """

    name = "SARIMA"

    def __init__(self, seasonality: Optional[int] = None, n_jobs: int = 1):
        self.seasonality = seasonality
        self.n_jobs = n_jobs

    @staticmethod
    def _fit_one(y, order, seasonal_order):
        import statsmodels.api as sm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return sm.tsa.statespace.SARIMAX(
                    y, order=order, seasonal_order=seasonal_order).fit(disp=False)
            except Exception:
                return None

    def fit(self, y_train, order=None) -> "SarimaForecaster":
        y, self.scale_ = _standardize(_as_1d(y_train))
        if order is None:
            from multioutreg.time_series.seasonality import detect_seasonal_period

            d = choose_d(y)
            s = self.seasonality
            if s is None:
                s = detect_seasonal_period(np.diff(y, n=d) if d else y) or 0
            seasonal = s >= 2 and y.size >= 2 * s + 2
            grid = [((p, d, q), (P, 0, Q, s) if seasonal else (0, 0, 0, 0))
                    for p in (0, 1) for q in (0, 1)
                    for P in ((0, 1) if seasonal else (0,))
                    for Q in ((0, 1) if seasonal else (0,))]
            fits = _parallel(self._fit_one, [(y, o, so) for o, so in grid], self.n_jobs)
            scored = [(r.aic, o, r) for o, r in zip(grid, fits)
                      if r is not None and np.isfinite(r.aic)]
            if not scored:
                raise RuntimeError("SARIMA: no (p, d, q)(P, D, Q, s) order could be fitted")
            _, order, self.results_ = min(scored, key=lambda t: t[0])
        else:
            self.results_ = self._fit_one(y, tuple(order[0]), tuple(order[1]))
            if self.results_ is None:
                raise RuntimeError(f"SARIMA{order} could not be fitted")
        self.order_ = (tuple(order[0]), tuple(order[1]))
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        mu, sd = self.scale_
        return mu + sd * np.asarray(self.results_.forecast(horizon), dtype=float)


class LstmForecaster:
    """Direct multi-step LSTM: ``train_window`` past values -> next ``horizon`` values.

    The scaler is fitted on the training series only, torch and numpy are
    seeded, and training uses shuffled mini-batches (the old per-sample loop
    took ~2 minutes for 150 points).
    """

    name = "LSTM"

    def __init__(self, horizon: int = 1, train_window: int = 10, epochs: int = 200,
                 batch_size: int = 16, lr: float = 0.005, random_state: int = 0):
        self.horizon = horizon
        self.train_window = train_window
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state

    def fit(self, y_train, order=None) -> "LstmForecaster":
        import torch
        from torch import nn
        from sklearn.preprocessing import MinMaxScaler

        from multioutreg.time_series.ts_dynamic_fit.algs.lstm import LSTMModel

        y = _as_1d(y_train)
        w, h = self.train_window, self.horizon
        n_seq = y.size - w - h + 1
        if n_seq < 1:
            raise ValueError(
                f"LSTM needs at least train_window + horizon = {w + h} training points, "
                f"got {y.size}")
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.scaler_ = MinMaxScaler(feature_range=(-1, 1)).fit(y.reshape(-1, 1))
        z = self.scaler_.transform(y.reshape(-1, 1)).ravel()
        X = torch.tensor(np.stack([z[i:i + w] for i in range(n_seq)]), dtype=torch.float32)
        Y = torch.tensor(np.stack([z[i + w:i + w + h] for i in range(n_seq)]), dtype=torch.float32)

        model = LSTMModel(output_size=h)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            perm = rng.permutation(n_seq)
            for start in range(0, n_seq, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        model.eval()
        self.model_ = model
        self.context_ = z[-w:]
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        import torch

        if horizon != self.horizon:
            raise ValueError(
                f"this LSTM was trained for horizon={self.horizon}, asked for {horizon}")
        with torch.no_grad():
            out = self.model_(torch.tensor(self.context_, dtype=torch.float32)).numpy()
        return self.scaler_.inverse_transform(out.reshape(-1, 1)).ravel()


def _parallel(fn, arg_list, n_jobs):
    if n_jobs == 1:
        return [fn(*args) for args in arg_list]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=n_jobs)(delayed(fn)(*args) for args in arg_list)
