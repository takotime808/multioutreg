# Copyright (c) 2026 takotime808
"""The public datasets in ``examples/example_data_ts/data/public`` and the script that builds them.

Checks that each CSV is a tidy ``date,<target>`` series with the documented
size and gaps, that seasonality detection finds the known period on the real
data, that ``fetch_public_datasets.py`` parses every source format (offline,
with small fake downloads), and that ``ts-pipeline`` runs end to end.
"""

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
TS_DIR = ROOT / "examples" / "example_data_ts"
DATA_DIR = TS_DIR / "data" / "public"

# file: (target, rows, first date, step, missing steps, seasonal period)
DATASETS = {
    "airpassengers_monthly.csv": ("passengers", 144, "1949-01-01", "MS", 0, 12),
    "co2_mauna_loa_monthly.csv": ("co2_ppm", 823, "1958-03-01", "MS", 0, 12),
    "melbourne_min_temp_daily.csv": ("min_temp_c", 3650, "1981-01-01", "D", 2, 365),
    "female_births_daily.csv": ("births", 365, "1959-01-01", "D", 0, 0),
    "sunspots_monthly.csv": ("sunspots", 2820, "1749-01-01", "MS", 0, 127),
    "bike_sharing_daily.csv": ("rentals", 731, "2011-01-01", "D", 0, 0),
    "bike_sharing_hourly.csv": ("rentals", 17379, "2011-01-01", "h", 165, 24),
    # Business days; bank holidays (~10 a year) are missing, so None: see below.
    "eurusd_daily.csv": ("usd_per_eur", 6974, "1999-01-04", "B", None, 0),
}
#: Series that gain rows when fetch_public_datasets.py is re-run.
GROWING = {"co2_mauna_loa_monthly.csv", "eurusd_daily.csv"}


def _load(name):
    return pd.read_csv(DATA_DIR / name, parse_dates=["date"])


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_is_a_tidy_series(name):
    target, rows, first, step, missing, _ = DATASETS[name]
    df = _load(name)
    assert list(df.columns) == ["date", target]
    assert pd.api.types.is_numeric_dtype(df[target]) and df[target].notna().all()
    assert df["date"].is_monotonic_increasing and df["date"].is_unique
    assert df["date"].iloc[0] == pd.Timestamp(first)
    assert len(df) >= rows if name in GROWING else len(df) == rows
    full = pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq=step)
    if missing is None:   # a trading calendar: holidays only, never weekends
        assert 0 < len(full) - len(df) <= 0.05 * len(full)
    else:
        assert len(full) - len(df) == missing
    assert set(df["date"]) <= set(full)


def test_readme_lists_every_dataset():
    readme = (DATA_DIR / "README.md").read_text()
    on_disk = {p.name for p in DATA_DIR.glob("*.csv")}
    assert on_disk == set(DATASETS)
    for name in DATASETS:
        assert f"`{name}`" in readme


@pytest.mark.parametrize("name", DATASETS)
def test_detects_the_known_season(name):
    from multioutreg.time_series.ts_dynamic_fit.src.evaluate import detect_seasonality

    target, *_, period = DATASETS[name]
    y = _load(name)[target].to_numpy(dtype=float)
    head = y[: len(y) - int(np.ceil(0.2 * len(y)))]   # what rank_forecasters looks at
    found = detect_seasonality(head)
    if period == 0:
        assert found == 0
    else:
        assert abs(found - period) <= 0.05 * period, found


# ---- fetch_public_datasets.py, offline ---------------------------------------

def _fake_downloads():
    hours = pd.DataFrame({"dteday": ["2011-01-01"] * 3 + ["2011-01-02"],
                          "hr": [0, 1, 3, 0], "cnt": [16, 40, 32, 17]})
    days = pd.DataFrame({"dteday": ["2011-01-01", "2011-01-02"], "cnt": [985, 801]})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("day.csv", days.to_csv(index=False))
        zf.writestr("hour.csv", hours.to_csv(index=False))
    return {
        # Bare-CR line endings, as the real AirPassengers file has.
        "airpassengers": b"Month,#Passengers\r1949-01,112\r1949-02,118\r",
        "co2": (b"# NOAA header\n# more comments\n"
                b"year,month,decimal date,average,deseasonalized,ndays,sdev,unc\n"
                b"1958,3,1958.2027,315.71,314.44,-1,-9.99,-0.99\n"
                b"1958,4,1958.2877,-99.99,315.16,-1,-9.99,-0.99\n"
                b"1958,5,1958.3699,317.51,314.69,-1,-9.99,-0.99\n"),
        "min_temp": b'"Date","Temp"\n"1981-01-02",17.9\n"1981-01-01",20.7\n',
        "births": b'\xef\xbb\xbf"Date","Births"\n"1959-01-01",35\n',
        "sunspots": b'"Month","Sunspots"\n"1749-01",58.0\n"1749-02",62.6\n',
        "bike": buf.getvalue(),
        # FRED leaves holidays blank; older exports wrote ".".
        "eurusd": (b"observation_date,DEXUSEU\n1999-01-04,1.1812\n1999-01-05,\n"
                   b"1999-01-06,.\n1999-01-07,1.1632\n"),
    }


def test_fetch_script_parses_every_source(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fetch_public", TS_DIR / "fetch_public_datasets.py")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)

    payloads = _fake_downloads()
    by_url = {fetch.URLS[k]: v for k, v in payloads.items()}
    monkeypatch.setattr(fetch, "_get", by_url.__getitem__)
    monkeypatch.setattr(fetch, "OUT_DIR", tmp_path)
    fetch.main()

    assert {p.name for p in tmp_path.glob("*.csv")} == set(DATASETS)
    read = lambda name: pd.read_csv(tmp_path / name)   # noqa: E731

    assert read("airpassengers_monthly.csv").to_dict("list") == {
        "date": ["1949-01-01", "1949-02-01"], "passengers": [112, 118]}
    # Comment lines skipped, the -99.99 sentinel month dropped.
    assert read("co2_mauna_loa_monthly.csv").to_dict("list") == {
        "date": ["1958-03-01", "1958-05-01"], "co2_ppm": [315.71, 317.51]}
    # Sorted by date.
    assert read("melbourne_min_temp_daily.csv")["date"].tolist() == ["1981-01-01", "1981-01-02"]
    # The UTF-8 BOM does not end up in the header.
    assert list(read("female_births_daily.csv").columns) == ["date", "births"]
    assert read("sunspots_monthly.csv")["date"].tolist() == ["1749-01-01", "1749-02-01"]
    assert read("bike_sharing_daily.csv")["rentals"].tolist() == [985, 801]
    # dteday + hr combined; the missing 02:00 stays missing.
    assert read("bike_sharing_hourly.csv").to_dict("list") == {
        "date": ["2011-01-01 00:00:00", "2011-01-01 01:00:00",
                 "2011-01-01 03:00:00", "2011-01-02 00:00:00"],
        "rentals": [16, 40, 32, 17]}
    # Both holiday markers dropped: the trading calendar is what remains.
    assert read("eurusd_daily.csv").to_dict("list") == {
        "date": ["1999-01-04", "1999-01-07"], "usd_per_eur": [1.1812, 1.1632]}


# ---- ts-pipeline end to end on real data ------------------------------------

def test_ts_pipeline_on_airpassengers(tmp_path):
    from typer.testing import CliRunner

    from mor_cli.ts_pipeline import app

    out = tmp_path / "out"
    res = CliRunner().invoke(app, [
        str(DATA_DIR / "airpassengers_monthly.csv"), "--datetime-col", "date",
        "--target-col", "passengers", "--freq", "MS", "--folds", "3",
        "--lstm-epochs", "2", "--out-dir", str(out)])
    assert res.exit_code == 0, res.output
    report = json.loads((out / "results.json").read_text())
    assert report["evaluation"]["horizon"] == 12
    mase = {m["Model"]: m["MASE"] for m in report["models"]}
    assert mase["SARIMA"] < mase["Naive"]
    assert len(report["forecast"]) == 12
    # 12-step forecasts from 3 origins about 8 apart: the Diebold-Mariano lag is
    # clipped, which used to be reported as "losses are identical".
    assert "identical" not in report["diebold_mariano"].get("reason", "")


def _finance_run(tmp_path, csv, *extra):
    from typer.testing import CliRunner

    from mor_cli.ts_pipeline import app

    out = tmp_path / "out"
    res = CliRunner().invoke(app, [
        str(csv), "--datetime-col", "date", "--target-col", "usd_per_eur",
        "--freq", "observed", "--fill", "ffill", "--no-clip-outliers",
        "--seasonality", "0", "--horizon", "5", "--folds", "3", "--lstm-epochs", "2",
        "--out-dir", str(out), *extra])
    return res, out


def test_ts_pipeline_on_eurusd_log_returns(tmp_path):
    prices = _load("eurusd_daily.csv").tail(300)
    csv = tmp_path / "eurusd.csv"
    prices.to_csv(csv, index=False)
    res, out = _finance_run(tmp_path, csv, "--transform", "log-return")
    assert res.exit_code == 0, res.output
    report = json.loads((out / "results.json").read_text())

    assert report["preprocessing"] == {"freq": "observed", "fill": "ffill",
                                       "clip_outliers": False, "transform": "log-return"}
    assert report["evaluation"]["horizon"] == 5
    # On returns the baseline is "no change", the random walk in prices.
    names = {m["Model"] for m in report["models"]}
    assert "Zero" in names and "Naive" not in names
    assert "diebold_mariano_baseline" in report
    # Origins are trading days from the file: no weekends were inserted.
    dates = set(prices["date"].astype(str))
    origins = pd.to_datetime(report["evaluation"]["origin_dates"])
    assert (origins.dayofweek < 5).all()
    assert {str(d.date()) for d in origins} <= dates
    # "forecast" is in log returns; "forecast_levels" is the same path as prices.
    r = np.asarray(report["forecast"])
    last = prices["usd_per_eur"].iloc[-1]
    assert np.abs(r).max() < 0.05
    np.testing.assert_allclose(report["forecast_levels"], last * np.exp(np.cumsum(r)))


def test_ts_pipeline_rejects_bad_finance_options(tmp_path):
    csv = tmp_path / "eurusd.csv"
    _load("eurusd_daily.csv").tail(100).to_csv(csv, index=False)
    res, _ = _finance_run(tmp_path, csv, "--transform", "pct")
    assert res.exit_code == 1 and "--transform must be one of" in res.output
    res, _ = _finance_run(tmp_path, csv, "--fill", "mean")
    assert res.exit_code == 1 and "--fill must be one of" in res.output

    neg = tmp_path / "neg.csv"
    pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=60),
                  "usd_per_eur": np.linspace(-1, 1, 60)}).to_csv(neg, index=False)
    res, _ = _finance_run(tmp_path, neg, "--transform", "log-return")
    assert res.exit_code == 1 and "strictly positive" in res.output
