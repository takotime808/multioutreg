# Copyright (c) 2026 takotime808
"""Download real, public time series and write them as tidy ``date,<target>`` CSVs.

The outputs in ``data/public/`` are inputs for the ``ts-pipeline`` CLI, e.g.::

    mor ts-pipeline examples/example_data_ts/data/public/airpassengers_monthly.csv \
        --datetime-col date --target-col passengers --freq MS

See ``data/public/README.md`` for sources, licences and suggested flags.
Re-run this script to refresh (Mauna Loa CO2 is updated monthly, EUR/USD daily)::

    python examples/example_data_ts/fetch_public_datasets.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "data" / "public"

BROWNLEE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
URLS = {
    "airpassengers": "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv",
    "co2": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv",
    "min_temp": f"{BROWNLEE}/daily-min-temperatures.csv",
    "births": f"{BROWNLEE}/daily-total-female-births.csv",
    "sunspots": f"{BROWNLEE}/monthly-sunspots.csv",
    "bike": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "eurusd": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU",
}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _csv(raw: bytes, **kwargs) -> pd.DataFrame:
    # AirPassengers ships with bare-CR line endings; normalise before parsing.
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    return pd.read_csv(io.StringIO(text), **kwargs)


def _write(df: pd.DataFrame, name: str) -> None:
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    path = OUT_DIR / name
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S" if "hourly" in name else "%Y-%m-%d")
    print(f"{path.name:32s} {len(df):6d} rows  {df['date'].iloc[0]:%Y-%m-%d} → {df['date'].iloc[-1]:%Y-%m-%d}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = _csv(_get(URLS["airpassengers"]))
    _write(df.set_axis(["date", "passengers"], axis=1), "airpassengers_monthly.csv")

    df = _csv(_get(URLS["co2"]), comment="#")
    df = pd.DataFrame({
        "date": pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)),
        # NOAA marks missing monthly means with negative sentinels.
        "co2_ppm": df["average"].where(df["average"] > 0),
    })
    _write(df, "co2_mauna_loa_monthly.csv")

    df = _csv(_get(URLS["min_temp"]))
    _write(df.set_axis(["date", "min_temp_c"], axis=1), "melbourne_min_temp_daily.csv")

    df = _csv(_get(URLS["births"]))
    _write(df.set_axis(["date", "births"], axis=1), "female_births_daily.csv")

    df = _csv(_get(URLS["sunspots"]))
    _write(df.set_axis(["date", "sunspots"], axis=1), "sunspots_monthly.csv")

    with zipfile.ZipFile(io.BytesIO(_get(URLS["bike"]))) as zf:
        day = pd.read_csv(zf.open("day.csv"))
        hour = pd.read_csv(zf.open("hour.csv"))
    _write(day.rename(columns={"dteday": "date", "cnt": "rentals"})[["date", "rentals"]],
           "bike_sharing_daily.csv")
    hour["date"] = pd.to_datetime(hour["dteday"]) + pd.to_timedelta(hour["hr"], unit="h")
    # hour.csv omits hours with no rentals, so the series has real gaps.
    _write(hour.rename(columns={"cnt": "rentals"})[["date", "rentals"]], "bike_sharing_hourly.csv")

    # FRED leaves bank holidays blank ("." in older exports); dropping them
    # keeps the trading calendar.
    df = _csv(_get(URLS["eurusd"]), na_values=["."])
    _write(df.set_axis(["date", "usd_per_eur"], axis=1), "eurusd_daily.csv")


if __name__ == "__main__":
    main()
