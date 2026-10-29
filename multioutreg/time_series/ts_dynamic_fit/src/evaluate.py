# Copyright (c) 2026 takotime808
"""Rank ARIMA / SARIMA / LSTM by rolling-origin, out-of-sample MASE (audit item 6).

The scheme is ``docs/AUDIT_CHECKLIST.md`` Appendix A.4:

1. Horizon H: ``horizon`` if given, else the seasonal period s when s >= 2,
   else 1.
2. Test region: the last ``max(H, ceil(0.2 n))`` points (FPP3 section 5.8).
3. Minimum training length ``max(2 s, train_window + H, 16)``; the test region
   shrinks toward H when the series is short, and a series shorter than
   ``min_train + H`` is rejected.
4. Origins: every origin in the test region (step 1) when there are at most
   :data:`MAX_ORIGINS`, else that many evenly spaced ones (``folds`` overrides).
5. Every model is refitted at every origin on all data before it and forecasts
   H steps on the original scale; all models are scored against the same
   values by one function. Rank on MASE averaged over origins (each origin
   scaled by its own training window).
6. A Diebold-Mariano test with the Harvey-Leybourne-Newbold correction
   compares the best and second-best model; it is reported, never used to
   pick the winner.

ARIMA / SARIMA search their order once, on the first (smallest) training
window, and only re-estimate parameters at later origins: the search sees no
test data, and it costs one search instead of one per origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from multioutreg.time_series.metrics import mase, smape

#: Most origins used; above this, origins are evenly spaced (a convention
#: borrowed from 10-fold CV; see AUDIT_CHECKLIST A.4).
MAX_ORIGINS = 10
#: Largest ARIMA parameter count in the default grid (p, q <= 7, plus a
#: constant and the variance) -- a training window must exceed it.
MIN_PARAM_POINTS = 16
#: ``baseline`` option -> the name of the model it adds.
BASELINES = {"naive": "Naive", "zero": "Zero"}


@dataclass(frozen=True)
class OriginPlan:
    """Where the models are refitted and scored."""
    n: int
    horizon: int
    seasonality: int                 # 0 when there is no season
    min_train: int
    test_region: int
    origins: tuple                   # index of the first forecast step of each origin
    scheme: str                      # "all origins" or "k evenly spaced origins"

    @property
    def mase_seasonality(self) -> int:
        return self.seasonality if self.seasonality >= 2 else 1

    def splits(self):
        """Yield ``(train_idx, test_idx)`` with an expanding training window."""
        for t in self.origins:
            yield np.arange(t), np.arange(t, t + self.horizon)

    def describe(self) -> str:
        return (f"{len(self.origins)} rolling origins ({self.scheme}) over the last "
                f"{self.test_region} points, horizon {self.horizon}, expanding window "
                f"from at least {self.min_train} points")


def plan_origins(
    n: int,
    seasonality: int = 0,
    horizon: Optional[int] = None,
    folds: Optional[int] = None,
    train_window: int = 10,
) -> OriginPlan:
    """Apply the A.4 rule to a series of length *n*."""
    s = int(seasonality or 0)
    h = int(horizon) if horizon else (s if s >= 2 else 1)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    min_train = max(2 * s, train_window + h, MIN_PARAM_POINTS)
    if n < min_train + h:
        raise ValueError(
            f"The series has {n} points; at least {min_train + h} are needed "
            f"({min_train} to train on and a horizon of {h} to test). Use a smaller "
            "--horizon or more data.")
    test_region = min(max(h, math.ceil(0.2 * n)), n - min_train)
    first, last = n - test_region, n - h
    m = last - first + 1
    k = min(folds, m) if folds else min(MAX_ORIGINS, m)
    if k >= m:
        origins = tuple(range(first, last + 1))
        scheme = "all origins"
    else:
        origins = tuple(sorted({int(round(t)) for t in np.linspace(first, last, k)}))
        scheme = f"{len(origins)} evenly spaced origins"
    return OriginPlan(n, h, s, min_train, test_region, origins, scheme)


def diebold_mariano(loss_a, loss_b, lag: int = 0) -> dict:
    """Diebold-Mariano test of equal expected loss, with the HLN correction.

    ``loss_a`` / ``loss_b`` are per-forecast losses (e.g. squared errors) of
    two models on the same targets; ``lag`` is the number of autocovariances
    in the long-run variance (``h - 1`` for overlapping h-step forecasts).
    Returns the HLN-corrected statistic (t with ``n - 1`` df) and two-sided
    p-value; a negative statistic means model a has the lower loss.
    Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
    """
    from scipy import stats

    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    if n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n}
    dc = d - d.mean()
    lag = max(0, min(int(lag), n - 1))
    gamma = [float(np.dot(dc[k:], dc[:n - k]) / n) for k in range(lag + 1)]
    var = gamma[0] + 2 * sum(gamma[1:])
    if not np.any(d):   # e.g. both models chose the same order
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n,
                "reason": "the two models' losses are identical"}
    if np.ptp(d) <= 1e-12 * np.abs(d).max():
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n,
                "reason": "the loss difference is the same at every origin"}
    if lag >= n - 1 or var <= 0:
        # At lag = n - 1 the autocovariances of the demeaned d sum to exactly 0
        # (rounding can leave a tiny positive var) and the HLN factor
        # (n - h)(n - h + 1) / n is 0; below it the estimate can still go
        # negative. Overlapping h-step forecasts need more origins than lags.
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n,
                "reason": (f"too few origins ({n}) to estimate the loss variance "
                           f"with {lag} autocovariance lag(s) from overlapping "
                           "forecasts; use more --folds or a shorter --horizon")}
    dm = d.mean() / math.sqrt(var / n)
    h = lag + 1
    hln = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 0.0))
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"statistic": float(stat), "p_value": float(p), "n": n}


@dataclass
class ModelScore:
    name: str
    mase: float
    rmse: float
    mae: float
    smape: float
    per_origin_mase: List[float]
    errors: np.ndarray               # (n_origins, horizon) forecast - actual
    order: object = None
    error: Optional[str] = None      # set when the model failed

    def as_dict(self) -> dict:
        return {"Model": self.name, "MASE": self.mase, "RMSE": self.rmse,
                "MAE": self.mae, "sMAPE": self.smape,
                "per_origin_MASE": self.per_origin_mase,
                "order": None if self.order is None else str(self.order),
                "error": self.error}


@dataclass
class RankingResult:
    plan: OriginPlan
    scores: List[ModelScore]         # successful models, best first
    failures: Dict[str, str]
    best_model: str
    diebold_mariano: Optional[dict] = None
    diebold_mariano_baseline: Optional[dict] = None   # winner vs the baseline
    final_model: object = None       # winner refitted on the full series
    forecast: Optional[np.ndarray] = None   # its next-H forecast
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "best_model": self.best_model,
            "evaluation": {
                "scheme": self.plan.describe(),
                "horizon": self.plan.horizon,
                "seasonality": self.plan.seasonality,
                "test_region": self.plan.test_region,
                "min_train": self.plan.min_train,
                "origins": list(self.plan.origins),
                "ranking_metric": "MASE (mean over origins)",
            },
            "models": [s.as_dict() for s in self.scores],
            "failures": self.failures,
            "diebold_mariano": self.diebold_mariano,
            "diebold_mariano_baseline": self.diebold_mariano_baseline,
            "forecast": None if self.forecast is None else [float(v) for v in self.forecast],
        }


def evaluate_forecasters(
    y,
    factories: Dict[str, Callable[[], object]],
    plan: OriginPlan,
    on_step: Optional[Callable[[str], None]] = None,
    refit_best: bool = True,
    baseline: Optional[str] = None,
) -> RankingResult:
    """Score every forecaster on *plan*'s origins and rank them by mean MASE.

    With ``baseline`` (a key of *factories*), the winner is also tested
    against that model -- the question that matters when the baseline is the
    random walk -- unless the baseline itself wins.
    """
    y = np.asarray(y, dtype=float).ravel()
    step = on_step or (lambda label: None)
    h, m = plan.horizon, plan.mase_seasonality
    scores, failures = [], {}
    for name, make in factories.items():
        step(f"Evaluating {name} on {len(plan.origins)} origins")
        try:
            order, errs, per_mase, smapes = None, [], [], []
            for tr, te in plan.splits():
                model = make().fit(y[tr], order=order)
                order = getattr(model, "order_", None)
                pred = np.asarray(model.forecast(h), dtype=float)
                if pred.shape != (h,) or not np.all(np.isfinite(pred)):
                    raise RuntimeError(f"forecast has shape {pred.shape} or non-finite values")
                errs.append(pred - y[te])
                per_mase.append(mase(y[te], pred, y[tr], seasonality=m))
                smapes.append(smape(y[te], pred))
            e = np.vstack(errs)
            scores.append(ModelScore(
                name=name, mase=float(np.mean(per_mase)),
                rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                smape=float(np.mean(smapes)), per_origin_mase=[float(v) for v in per_mase],
                errors=e, order=order))
        except Exception as exc:
            failures[name] = str(exc)
    if not scores:
        raise RuntimeError("All models failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))
    scores.sort(key=lambda s: (s.mase, s.rmse))

    gaps = np.diff(plan.origins)
    min_gap = int(gaps.min()) if gaps.size else h

    def dm_test(a: ModelScore, b: ModelScore) -> dict:
        out = diebold_mariano(
            (a.errors ** 2).mean(axis=1), (b.errors ** 2).mean(axis=1),
            lag=max(0, h - min_gap))
        out.update({"model_a": a.name, "model_b": b.name, "loss": "mean squared error per origin",
                    "note": "reported only; does not decide the winner"})
        return out

    dm = dm_test(scores[0], scores[1]) if len(scores) >= 2 else None
    by_name = {s.name: s for s in scores}
    dm_baseline = None
    if baseline in by_name and scores[0].name != baseline:
        dm_baseline = dm_test(scores[0], by_name[baseline])

    result = RankingResult(plan, scores, failures, scores[0].name, dm, dm_baseline)
    if refit_best:
        step(f"Refitting {result.best_model} on the full series")
        result.final_model = factories[result.best_model]().fit(y)
        result.forecast = np.asarray(result.final_model.forecast(h), dtype=float)
    return result


def default_factories(plan: OriginPlan, train_window: int = 10, lstm_epochs: int = 100,
                      n_jobs: int = -1, include_naive: bool = True,
                      baseline: str = "naive") -> Dict[str, Callable]:
    """ARIMA, SARIMA, LSTM (and a baseline) configured for *plan*.

    ``baseline='naive'`` adds the (seasonal) last-value forecast; ``'zero'``
    adds the no-change forecast, the right benchmark for a series of returns
    or differences (it is the random walk in levels).
    """
    from multioutreg.time_series.ts_dynamic_fit.src.forecasters import (
        ArimaForecaster, LstmForecaster, NaiveForecaster, SarimaForecaster, ZeroForecaster)

    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {tuple(BASELINES)}, got {baseline!r}")

    s = plan.seasonality if plan.seasonality >= 2 else None
    factories = {
        "ARIMA": lambda: ArimaForecaster(n_jobs=n_jobs),
        "SARIMA": lambda: SarimaForecaster(seasonality=s, n_jobs=n_jobs),
        "LSTM": lambda: LstmForecaster(horizon=plan.horizon, train_window=train_window,
                                       epochs=lstm_epochs),
    }
    if include_naive and baseline == "zero":
        factories["Zero"] = ZeroForecaster
    elif include_naive:
        factories["Naive"] = lambda: NaiveForecaster(seasonality=plan.seasonality)
    return factories


def rank_forecasters(
    y,
    horizon: Optional[int] = None,
    folds: Optional[int] = None,
    seasonality: Optional[int] = None,
    train_window: int = 10,
    lstm_epochs: int = 100,
    n_jobs: int = -1,
    on_step: Optional[Callable[[str], None]] = None,
    baseline: str = "naive",
) -> RankingResult:
    """Plan the origins, evaluate ARIMA / SARIMA / LSTM / a baseline, rank, refit the winner.

    The seasonal period is detected on the first 80% of the series only (the
    part that is never a test target), unless ``seasonality`` is given.
    ``baseline`` is ``'naive'`` or ``'zero'`` (see :func:`default_factories`).
    """
    y = np.asarray(y, dtype=float).ravel()
    if not np.all(np.isfinite(y)):
        raise ValueError("the series contains missing or non-finite values")
    n = y.size
    if seasonality is None:
        head = y[: n - math.ceil(0.2 * n)]
        seasonality = detect_seasonality(head) if head.size >= 8 else 0
    plan = plan_origins(n, seasonality, horizon, folds, train_window)
    factories = default_factories(plan, train_window=train_window,
                                  lstm_epochs=lstm_epochs, n_jobs=n_jobs, baseline=baseline)
    return evaluate_forecasters(y, factories, plan, on_step=on_step,
                                baseline=BASELINES[baseline])


def detect_seasonality(y) -> int:
    """Seasonal period of *y* (after the ADF-chosen differencing), 0 if none."""
    from multioutreg.time_series.seasonality import detect_seasonal_period
    from multioutreg.time_series.ts_dynamic_fit.src.forecasters import choose_d

    y = np.asarray(y, dtype=float).ravel()
    d = choose_d(y)
    return int(detect_seasonal_period(np.diff(y, n=d) if d else y) or 0)
