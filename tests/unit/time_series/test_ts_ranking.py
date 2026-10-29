# Copyright (c) 2026 takotime808
"""Rolling-origin, original-scale ranking of ARIMA / SARIMA / LSTM (audit item 6, A.6)."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit.src.evaluate import (
    MAX_ORIGINS,
    detect_seasonality,
    diebold_mariano,
    evaluate_forecasters,
    plan_origins,
)
from multioutreg.time_series.ts_dynamic_fit.src.forecasters import (
    ArimaForecaster,
    LstmForecaster,
    NaiveForecaster,
    SarimaForecaster,
    choose_d,
)


def _trend(n=120, seed=0, level=1000.0):
    rng = np.random.default_rng(seed)
    return level + np.cumsum(2 + rng.normal(0, 5, n))


def _ar1(n=150, phi=0.8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + rng.normal()
    return y + 50


def _fast_factories(plan):
    return {
        "ARIMA": lambda: ArimaForecaster(max_order=2),
        "Naive": lambda: NaiveForecaster(seasonality=plan.seasonality),
    }


# ---- origin plan (A.4) -------------------------------------------------

def test_plan_defaults_without_season():
    plan = plan_origins(100)
    assert plan.horizon == 1 and plan.min_train == 16
    assert plan.test_region == 20                  # ceil(0.2 n)
    assert len(plan.origins) == MAX_ORIGINS        # 20 possible > 10 -> evenly spaced
    assert plan.origins[0] == 80 and plan.origins[-1] == 99
    assert "evenly spaced" in plan.scheme


def test_plan_uses_every_origin_when_few():
    plan = plan_origins(40)                        # test region 8 -> 8 origins
    assert plan.origins == tuple(range(32, 40)) and plan.scheme == "all origins"


def test_plan_seasonal_horizon_and_min_train():
    plan = plan_origins(120, seasonality=7)
    assert plan.horizon == 7 and plan.min_train == 17   # max(14, 10 + 7, 16)
    assert plan.origins[-1] == 120 - 7
    for tr, te in plan.splits():
        assert len(tr) >= plan.min_train and len(te) == 7 and te[-1] < 120


def test_plan_folds_override_and_short_series():
    assert len(plan_origins(100, folds=3).origins) == 3
    assert len(plan_origins(100, horizon=5, folds=50).origins) == 16  # all 16 possible
    # The test region shrinks toward H before giving up.
    plan = plan_origins(20)
    assert plan.min_train == 16 and plan.test_region == 4
    with pytest.raises(ValueError, match="at least 17"):
        plan_origins(16)


# ---- A.6 tests ------------------------------------------------------------

def test_every_model_scored_on_the_same_targets():
    y = _trend()
    plan = plan_origins(len(y))
    res = evaluate_forecasters(y, _fast_factories(plan), plan, refit_best=False)
    for score in res.scores:
        assert score.errors.shape == (len(plan.origins), plan.horizon)
    # Errors are forecast - actual on the original scale: differences of two
    # models' errors are differences of their forecasts, never of the targets.
    naive = next(s for s in res.scores if s.name == "Naive")
    expected = np.array([y[t - 1] - y[t] for t in plan.origins]).reshape(-1, 1)
    np.testing.assert_allclose(naive.errors, expected)


def test_ranking_is_scale_invariant():
    y = _trend()
    plan = plan_origins(len(y))
    a = evaluate_forecasters(y, _fast_factories(plan), plan, refit_best=False)
    b = evaluate_forecasters(1000 * y, _fast_factories(plan), plan, refit_best=False)
    assert a.best_model == b.best_model
    for sa, sb in zip(a.scores, b.scores):
        assert sa.name == sb.name
        # Standardising inside ARIMA makes the fits agree up to optimiser
        # round-off (before it, x1000 moved ARIMA's MASE by ~10%).
        assert sa.mase == pytest.approx(sb.mase, rel=1e-2)


def test_order_search_never_sees_test_values():
    y = _trend()
    plan = plan_origins(len(y))
    first = plan.origins[0]
    tampered = y.copy()
    tampered[first:] += np.linspace(0, 500, len(y) - first)
    o1 = ArimaForecaster(max_order=2).fit(y[:first]).order_
    o2 = ArimaForecaster(max_order=2).fit(tampered[:first]).order_
    assert o1 == o2
    # ...and the forecast for the first origin only depends on data before it.
    r1 = evaluate_forecasters(y, _fast_factories(plan), plan, refit_best=False)
    r2 = evaluate_forecasters(tampered, _fast_factories(plan), plan, refit_best=False)
    for s1, s2 in zip(r1.scores, r2.scores):
        f1 = s1.errors[0] + y[first:first + plan.horizon]
        f2 = s2.errors[0] + tampered[first:first + plan.horizon]
        np.testing.assert_allclose(f1, f2)


def test_arima_beats_naive_on_ar1():
    y = _ar1()
    plan = plan_origins(len(y), folds=10)
    res = evaluate_forecasters(y, _fast_factories(plan), plan, refit_best=False)
    by_name = {s.name: s for s in res.scores}
    assert by_name["ARIMA"].mase < by_name["Naive"].mase
    assert res.best_model == "ARIMA"


def test_seasonal_series_detects_period_and_sarima_uses_it():
    rng = np.random.default_rng(1)
    t = np.arange(140)
    y = 20 + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.3, t.size)
    assert detect_seasonality(y) == 7
    model = SarimaForecaster().fit(y)
    assert model.order_[1][3] == 7
    plan = plan_origins(len(y), seasonality=7)
    assert plan.horizon == 7


def test_sarima_skips_seasons_longer_than_the_cap():
    from multioutreg.time_series.ts_dynamic_fit.src.forecasters import SARIMA_MAX_SEASON

    rng = np.random.default_rng(1)
    s = SARIMA_MAX_SEASON + 1
    t = np.arange(3 * s)
    y = 20 + 5 * np.sin(2 * np.pi * t / s) + rng.normal(0, 0.3, t.size)
    model = SarimaForecaster(seasonality=s).fit(y)
    assert model.order_[1] == (0, 0, 0, 0)
    assert model.forecast(s).shape == (s,)


def test_choose_d():
    rng = np.random.default_rng(0)
    assert choose_d(rng.normal(size=200)) == 0
    assert choose_d(np.cumsum(rng.normal(size=200))) == 1
    assert choose_d(np.full(50, 3.0)) == 0
    assert choose_d(np.cumsum(np.cumsum(rng.normal(size=300)))) <= 2


def test_lstm_forecaster_is_seeded_and_on_original_scale():
    y = _trend(n=60)
    a = LstmForecaster(horizon=3, epochs=3).fit(y).forecast(3)
    b = LstmForecaster(horizon=3, epochs=3).fit(y).forecast(3)
    np.testing.assert_allclose(a, b)
    assert a.shape == (3,) and y.min() - 200 < a.mean() < y.max() + 200
    with pytest.raises(ValueError, match="horizon"):
        LstmForecaster(horizon=3, epochs=1).fit(y).forecast(4)
    with pytest.raises(ValueError, match="at least"):
        LstmForecaster(horizon=5, train_window=10).fit(y[:12])


def test_failed_model_is_reported_not_fatal():
    y = _trend()
    plan = plan_origins(len(y))

    class Broken:
        def fit(self, y, order=None):
            raise RuntimeError("boom")

    factories = {**_fast_factories(plan), "Broken": Broken}
    res = evaluate_forecasters(y, factories, plan, refit_best=False)
    assert res.failures == {"Broken": "boom"}
    assert {s.name for s in res.scores} == {"ARIMA", "Naive"}


def test_zero_baseline_for_returns():
    from multioutreg.time_series.ts_dynamic_fit.src.evaluate import default_factories
    from multioutreg.time_series.ts_dynamic_fit.src.forecasters import ZeroForecaster

    assert np.array_equal(ZeroForecaster().fit([1.0, 2.0]).forecast(3), np.zeros(3))
    plan = plan_origins(200)
    assert "Zero" in default_factories(plan, baseline="zero")
    assert "Naive" not in default_factories(plan, baseline="zero")
    assert "Naive" in default_factories(plan)
    with pytest.raises(ValueError, match="baseline must be one of"):
        default_factories(plan, baseline="mean")

    # i.i.d. returns: 0 is the best forecast, and "the last return repeats"
    # (Naive) has about twice its squared error.
    r = np.random.default_rng(0).normal(0, 0.01, 400)
    plan = plan_origins(r.size, folds=40)
    factories = {"Naive": lambda: NaiveForecaster(), "Zero": ZeroForecaster}
    res = evaluate_forecasters(r, factories, plan, refit_best=False)
    by = {s.name: s for s in res.scores}
    assert 1.5 < (by["Naive"].rmse / by["Zero"].rmse) ** 2 < 2.5


def test_winner_is_tested_against_the_baseline():
    y = _ar1(n=200)
    plan = plan_origins(len(y), folds=20)
    class Mean:
        def fit(self, y, order=None):
            self.mu = float(np.mean(y))
            return self

        def forecast(self, horizon):
            return np.full(horizon, self.mu)

    factories = {**_fast_factories(plan), "Mean": Mean}
    res = evaluate_forecasters(y, factories, plan, refit_best=False, baseline="Naive")
    assert res.best_model == "ARIMA"
    dmb = res.diebold_mariano_baseline
    assert (dmb["model_a"], dmb["model_b"]) == ("ARIMA", "Naive")
    assert res.as_dict()["diebold_mariano_baseline"] == dmb
    # No baseline, or the baseline wins: nothing to test.
    assert evaluate_forecasters(y, factories, plan, refit_best=False).diebold_mariano_baseline is None
    only = {"Naive": factories["Naive"]}
    assert evaluate_forecasters(y, only, plan, refit_best=False,
                                baseline="Naive").diebold_mariano_baseline is None


def test_winner_is_refitted_on_full_series():
    y = _trend()
    plan = plan_origins(len(y))
    res = evaluate_forecasters(y, _fast_factories(plan), plan)
    assert res.forecast.shape == (plan.horizon,)
    out = res.as_dict()
    json.dumps(out)
    assert out["best_model"] == res.scores[0].name
    assert out["evaluation"]["origins"] == list(plan.origins)


# ---- Diebold-Mariano -------------------------------------------------------

def test_diebold_mariano():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0, 1, 200) ** 2, rng.normal(0, 2, 200) ** 2
    out = diebold_mariano(good, bad)
    assert out["statistic"] < 0 and out["p_value"] < 0.01
    same = diebold_mariano(good, good)
    assert np.isnan(same["p_value"]) and "identical" in same["reason"]
    shifted = diebold_mariano(good, good + 1.0)
    assert np.isnan(shifted["p_value"]) and "same at every origin" in shifted["reason"]
    assert np.isnan(diebold_mariano([1.0], [2.0])["p_value"])
    # HLN shrinks the statistic toward 0 for multi-step (lagged) losses.
    a = diebold_mariano(good[:20], bad[:20], lag=0)["statistic"]
    b = diebold_mariano(good[:20], bad[:20], lag=2)["statistic"]
    assert np.isfinite(a) and np.isfinite(b)


def test_diebold_mariano_clipped_lag_is_not_called_identical():
    # 5 origins of 12-step forecasts 4-5 points apart (AirPassengers): lag 8 is
    # clipped to n - 1 = 4, where the long-run variance is exactly 0. That was
    # reported as "losses are identical" for SARIMA vs Naive, MAE 20 vs 45.
    sarima = np.array([0.41, 1.83, 0.19, 0.17, 0.32])
    naive = np.array([0.94, 2.10, 3.18, 3.33, 2.47])
    out = diebold_mariano(sarima, naive, lag=8)
    assert np.isnan(out["p_value"])
    assert "identical" not in out["reason"] and "too few origins" in out["reason"]
    # Rounding can leave the lag n - 1 variance slightly positive, and the HLN
    # factor there is 0: that printed stat=-0.000, p=1.000 on sunspots.
    rng = np.random.default_rng(0)
    for n in (2, 3, 5, 8):
        for _ in range(50):
            out = diebold_mariano(rng.random(n), rng.random(n) + 0.5, lag=n - 1)
            assert np.isnan(out["statistic"]) and "too few origins" in out["reason"]


# ---- legacy run() fixes (items 23, 24, A.3) ------------------------------

def test_legacy_sarima_diff_count_zero_on_white_noise():
    from multioutreg.time_series.ts_dynamic_fit import SARIMA

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(10, 1, 80)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta, _, _ = SARIMA(df, "y").run()
    assert meta["metadata"]["diff_count"] == 0


def test_legacy_arima_mape_is_mean_absolute():
    from multioutreg.time_series.ts_dynamic_fit import ARIMA

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": 10 + rng.normal(0, 1, 60)})
    model = ARIMA(df, "y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta, data, fitted = model.run()
    pred = np.asarray(fitted.predict())
    actual = np.asarray(data)
    mask = actual != 0
    expected = np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask]))
    assert meta["performance"]["MAPE"] == pytest.approx(expected)


# ---- CLI smoke test (A.6 step 6) --------------------------------------------

def test_ts_pipeline_cli(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from mor_cli.ts_pipeline import app

    y = _trend(n=120)
    csv = tmp_path / "series.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=120, freq="D"),
                  "y": y}).to_csv(csv, index=False)
    out = tmp_path / "out"
    res = CliRunner().invoke(app, [str(csv), "--target-col", "y", "--datetime-col", "date",
                                   "--lstm-epochs", "2", "--folds", "3", "--out-dir", str(out)])
    assert res.exit_code == 0, res.output
    report = json.loads((out / "results.json").read_text())
    assert report["evaluation"]["horizon"] == 1
    assert len(report["evaluation"]["origins"]) == 3
    assert len(report["evaluation"]["origin_dates"]) == 3
    assert {m["Model"] for m in report["models"]} >= {"ARIMA", "SARIMA", "Naive"}
    assert report["best_model"] in {m["Model"] for m in report["models"]}

    short = tmp_path / "short.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=12, freq="D"),
                  "y": y[:12]}).to_csv(short, index=False)
    res = CliRunner().invoke(app, [str(short), "--target-col", "y", "--datetime-col", "date",
                                   "--out-dir", str(tmp_path / "out2")])
    assert res.exit_code == 1
    assert "at least" in res.output
