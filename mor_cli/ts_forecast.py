# Copyright (c) 2025 takotime808
"""CLI for zero-shot forecasting with Chronos / Chronos-Bolt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import pandas as pd
import typer

from multioutreg.time_series.chronos_adapter import ChronosForecaster

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _parse_quantiles(text: str) -> List[float]:
    qs = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not qs:
        raise typer.BadParameter("Provide at least one quantile, e.g. '0.1,0.5,0.9'.")
    for q in qs:
        if not (0.0 < q < 1.0):
            raise typer.BadParameter(f"Quantiles must be in (0,1); got {q}.")
    return qs


@app.command(name="ts-forecast")
def ts_forecast(
    csv: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV file containing the time series.",
    ),
    time_col: Optional[str] = typer.Option(
        None, "--time-col", help="Timestamp column (optional; ignored for modeling)."
    ),
    value_cols: Optional[str] = typer.Option(
        None,
        "--value-cols",
        help="Comma-separated target columns. If omitted, uses the first numeric column.",
    ),
    horizon: int = typer.Option(
        ...,
        "--horizon",
        min=1,
        help="Prediction horizon (number of future steps).",
    ),
    model: str = typer.Option(
        "amazon/chronos-bolt-base",
        "--model",
        help="Chronos model name (e.g., amazon/chronos-bolt-base).",
    ),
    quantiles: str = typer.Option(
        "0.1,0.5,0.9",
        "--quantiles",
        help="Comma-separated quantiles to output, e.g. '0.05,0.5,0.95'.",
    ),
    out: Path = typer.Option(
        Path("forecast.csv"),
        "--out",
        file_okay=True,
        dir_okay=False,
        writable=True,
        help="Output CSV path for tidy long format (id, step, quantile, value).",
    ),
):
    """Generate zero-shot probabilistic forecasts and save them to CSV."""
    # Load data
    df = pd.read_csv(csv)

    # Resolve target columns
    if value_cols:
        cols = [c.strip() for c in value_cols.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise typer.BadParameter(f"Missing columns in CSV: {missing}")
    else:
        numeric = [
            c for c in df.columns
            if (time_col is None or c != time_col) and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not numeric:
            raise typer.BadParameter("No numeric columns found to use as targets.")
        cols = numeric[:1]  # default to first numeric column

    # Build series dict
    series = {c: df[c].dropna().to_numpy() for c in cols}

    # Parse quantiles
    q_levels = tuple(_parse_quantiles(quantiles))

    # Forecast
    forecaster = ChronosForecaster(model_name=model)
    forecaster.fit(series)
    res = forecaster.predict(prediction_length=horizon, quantiles=q_levels)

    # Write tidy long CSV: id, step, quantile, value
    rows = []
    ids = res.ids or tuple(f"y{i}" for i in range(res.quantiles.shape[0]))
    for i, sid in enumerate(ids):
        for qi, q in enumerate(res.q_levels):
            for h in range(horizon):
                rows.append(
                    {
                        "id": sid,
                        "step": h + 1,
                        "quantile": q,
                        "value": float(res.quantiles[i, qi, h]),
                    }
                )
    pd.DataFrame(rows).to_csv(out, index=False)

    typer.echo(json.dumps({"written": str(out), "ids": list(ids), "quantiles": list(res.q_levels)}))


if __name__ == "__main__":
    app()
