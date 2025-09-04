# Copyright (c) 2025 takotime808

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd

from multioutreg.time_series.chronos_adapter import (
    ChronosForecaster,
    ForecastResult,
)


def load_financial_csv(path: str | Path, column: str = "Close") -> pd.Series:
    """Load a financial time series from a CSV file.

    The CSV is expected to have a date column as the first column and the
    specified price column. Dates are parsed and used as the index.

    Parameters
    ----------
    path:
        File path to the CSV file.
    column:
        Name of the column to extract. Default is ``"Close"``.
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    if column not in df.columns:
        raise KeyError(f"{column!r} not found in CSV columns {df.columns.tolist()}")
    return df[column].astype(float)


def forecast_with_chronos(
    series: pd.Series,
    horizon: int,
    forecaster: Optional[ChronosForecaster] = None,
    **kwargs,
) -> ForecastResult:
    """Forecast a financial series using :class:`ChronosForecaster`.

    Parameters
    ----------
    series:
        Historical price series.
    horizon:
        Number of future steps to forecast.
    forecaster:
        Optional pre-configured ``ChronosForecaster``. If ``None``, one is
        created using ``kwargs``.
    **kwargs:
        Additional keyword arguments passed to ``ChronosForecaster`` when the
        ``forecaster`` is ``None``.
    """
    model = forecaster or ChronosForecaster(**kwargs)
    model.fit(series)
    return model.predict(prediction_length=int(horizon))