# Copyright (c) 2026 takotime808
"""CLI for ranking ARIMA / SARIMA / LSTM time series models via the ts_dynamic_fit pipeline.

Models are scored out-of-sample, on the original scale, at rolling forecast
origins and ranked by MASE (audit item 6; see ``src/evaluate.py``).
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command(name="ts-pipeline")
def ts_pipeline(
    csv: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="CSV file containing time series data.",
    ),
    target_col: str = typer.Option(
        ...,
        "--target-col",
        help="Name of the target variable column to forecast.",
    ),
    datetime_col: Optional[str] = typer.Option(
        None,
        "--datetime-col",
        help="Name of the datetime column. Omit to treat the row index as the time axis.",
    ),
    freq: str = typer.Option(
        "1D",
        "--freq",
        help="Resampling frequency passed to DataProcessor (e.g. 1D, 1H, 15min, 1W).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output from DataProcessor and model training.",
    ),
    out_dir: Path = typer.Option(
        Path("ts_pipeline_output"),
        "--out-dir",
        file_okay=False,
        writable=True,
        help="Directory where results JSON and model artifacts are written.",
    ),
    agg: str = typer.Option(
        "sum",
        "--agg",
        help=(
            "Aggregation function applied when multiple rows share the same timestamp "
            "(e.g. order-level data). Choices: sum, mean, median, last, first."
        ),
    ),
    horizon: Optional[int] = typer.Option(
        None,
        "--horizon",
        min=1,
        help="Forecast horizon H to score. Default: the seasonal period if one is detected, else 1.",
    ),
    folds: Optional[int] = typer.Option(
        None,
        "--folds",
        min=1,
        help="Number of rolling forecast origins. Default: all origins up to 10, else 10 evenly spaced.",
    ),
    lstm_epochs: int = typer.Option(
        100,
        "--lstm-epochs",
        min=1,
        help="LSTM training epochs per fit (the LSTM is refitted at every origin).",
    ),
) -> None:
    """Rank ARIMA, SARIMA, LSTM and a naive baseline out-of-sample; forecast with the best."""

    # Deferred imports — same pattern as ts_forecast.py
    try:
        from multioutreg.time_series.ts_dynamic_fit import DataProcessor
        from multioutreg.time_series.ts_dynamic_fit.src.evaluate import rank_forecasters
    except Exception as exc:
        typer.echo(
            f"Error: could not import ts_dynamic_fit modules. "
            f"Ensure statsmodels, torch, and joblib are installed.\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    csv = csv.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    df = pd.read_csv(csv)

    if target_col not in df.columns:
        typer.echo(f"Error: column '{target_col}' not found in {csv}.", err=True)
        raise typer.Exit(code=1)

    # ── Coerce target column to numeric ───────────────────────
    # Handle percentage strings (e.g. "19.30%") and other formatted numbers.
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        cleaned = df[target_col].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip()
        df[target_col] = pd.to_numeric(cleaned, errors="coerce")
        typer.echo(f"Converted '{target_col}' from string to numeric (stripped %, commas).")

    # ── Pre-aggregate if datetime column has duplicate values ──
    # (e.g. order-level CSVs with one row per item, multiple per day)
    dt_arg = datetime_col or df.columns[0]
    if dt_arg in df.columns:
        df[dt_arg] = pd.to_datetime(df[dt_arg])
        if df[dt_arg].duplicated().any():
            typer.echo(
                f"Duplicate timestamps detected in '{dt_arg}'. "
                f"Aggregating '{target_col}' by {freq} using '{agg}'..."
            )
            agg_df = (
                df.groupby(pd.Grouper(key=dt_arg, freq=freq))[target_col]
                .agg(agg)
                .reset_index()
            )
            df = agg_df

    # ── DataProcessor ─────────────────────────────────────────
    processor = DataProcessor(verbose=verbose)
    processor.load_data(df)

    processor.validate_data(datetime_col=dt_arg, target_col=target_col)
    processed_data = processor.preprocess_data(
        datetime_col=dt_arg,
        target_col=target_col,
        freq=freq,
    )

    # ── Rank models out-of-sample ─────────────────────────────
    series = processed_data[target_col]
    try:
        ranking = rank_forecasters(
            series.to_numpy(dtype=float),
            horizon=horizon,
            folds=folds,
            lstm_epochs=lstm_epochs,
            on_step=typer.echo,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    for label, msg in ranking.failures.items():
        typer.echo(f"Warning: {label} failed — {msg}", err=True)
    typer.echo(f"Evaluation: {ranking.plan.describe()}")
    for score in ranking.scores:
        typer.echo(f"  {score.name:7s} MASE={score.mase:.4f}  RMSE={score.rmse:.4g}  "
                   f"MAE={score.mae:.4g}  sMAPE={score.smape:.2f}")
    dm = ranking.diebold_mariano
    if dm and dm.get("reason"):
        typer.echo(f"Diebold-Mariano ({dm['model_a']} vs {dm['model_b']}): not computed, "
                   f"{dm['reason']}")
    elif dm and dm.get("n", 0) >= 2:
        typer.echo(f"Diebold-Mariano ({dm['model_a']} vs {dm['model_b']}): "
                   f"stat={dm['statistic']:.3f}, p={dm['p_value']:.3f} (reported only)")

    # ── Write results ─────────────────────────────────────────
    report = ranking.as_dict()
    index = series.index
    report["evaluation"]["origin_dates"] = [str(index[t]) for t in ranking.plan.origins]
    report["evaluation"]["train_start"] = str(index[0])
    report["evaluation"]["last_observation"] = str(index[-1])

    results_path = out_dir / "results.json"
    results_path.write_text(json.dumps(report, indent=2))

    typer.echo(
        json.dumps(
            {"best_model": ranking.best_model, "out_dir": str(out_dir.resolve())},
        )
    )


if __name__ == "__main__":
    app()
