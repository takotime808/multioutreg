# Copyright (c) 2026 takotime808
"""CLI for training ARIMA / SARIMA / LSTM time series models via the ts_dynamic_fit pipeline."""

from __future__ import annotations

import json
import os as _os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

# ------------------------------------------------------------------
# sys.path injection — ts_dynamic_fit uses bare relative imports
# that require its directory to be on the Python path.
# ------------------------------------------------------------------
_TS_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "../multioutreg/time_series/ts_dynamic_fit")
)
if _TS_DIR not in sys.path:
    sys.path.insert(0, _TS_DIR)

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
) -> None:
    """Train and compare ARIMA, SARIMA, and LSTM on a CSV file; save the best model."""

    # Deferred imports — same pattern as ts_forecast.py
    try:
        from data_handling.DataProcessor import DataProcessor
        from algs.arima import ARIMA
        from algs.sarima import SARIMA
        from algs.lstm import LSTM
        from src.ranker import Ranker
    except Exception as exc:
        typer.echo(
            f"Error: could not import ts_dynamic_fit modules. "
            f"Ensure statsmodels, torch, and joblib are installed.\n{exc}",
            err=True,
        )
        raise typer.Exit(code=1)

    # Resolve absolute path before any chdir so relative paths keep working
    csv = csv.resolve()

    # ── Setup output directory and CWD ────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    (out_dir / "temp").mkdir(exist_ok=True)

    original_cwd = _os.getcwd()
    _os.chdir(out_dir)

    try:
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
        _, future_pred = processor.load_data(df)

        processor.validate_data(datetime_col=dt_arg, target_col=target_col)
        processed_data = processor.preprocess_data(
            datetime_col=dt_arg,
            target_col=target_col,
            freq=freq,
        )

        # ── Train models ──────────────────────────────────────────
        results: list[tuple] = []

        for ModelClass, label, kwargs in [
            (ARIMA, "ARIMA", {"verbose": verbose}),
            (SARIMA, "SARIMA", {"verbose": verbose}),
            (LSTM, "LSTM", {"fut_pred": future_pred, "train_window": 10, "verbose": verbose}),
        ]:
            typer.echo(f"Training {label}...")
            try:
                if label == "LSTM":
                    m = ModelClass(processed_data, feature_column=target_col, **kwargs)
                else:
                    m = ModelClass(processed_data, target_col, **kwargs)
                perf, data_obj, model_obj = m.run()
                results.append((perf, data_obj, model_obj))
            except Exception as exc:
                typer.echo(f"Warning: {label} failed — {exc}", err=True)

        if not results:
            typer.echo("Error: all models failed. Check data and column selection.", err=True)
            raise typer.Exit(code=1)

        # ── Rank and select best ───────────────────────────────────
        perf_dicts = [r[0] for r in results]
        best_model_name = Ranker(df=pd.DataFrame(perf_dicts), verbose=verbose).get_best()

        # ── Write results ─────────────────────────────────────────
        def _make_json_safe(obj):
            """Recursively convert non-JSON-serializable keys/values to strings."""
            if isinstance(obj, dict):
                return {str(k): _make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_json_safe(i) for i in obj]
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        results_path = Path("results.json")
        results_path.write_text(
            json.dumps(
                _make_json_safe({"best_model": best_model_name, "models": [r[0] for r in results]}),
                indent=2,
            )
        )

        typer.echo(
            json.dumps(
                {"best_model": best_model_name, "out_dir": str(out_dir.resolve())},
            )
        )

    finally:
        _os.chdir(original_cwd)


if __name__ == "__main__":
    app()
