# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows
import typer
import pandas as pd
from typing import Optional
from typing_extensions import Annotated
from multioutreg.sampling.detect import detect_sampling_method

# Arg = typer.Argument
# Opt = typer.Option
app = typer.Typer(
    name="multioutreg",
    rich_markup_mode="rich",
)


def _split(names: Optional[str]) -> list[str]:
    return [c.strip() for c in names.split(",") if c.strip()] if names else []


@app.command(
    "detect_sampling",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def detect_sampling(
    file_path: Annotated[str, typer.Argument(..., help="Path to CSV dataset")],
    columns: Annotated[Optional[str], typer.Option(
        "--columns", help="Comma-separated input columns to test (default: all numeric columns)")] = None,
    exclude: Annotated[Optional[str], typer.Option(
        "--exclude", help="Comma-separated columns to leave out, e.g. the responses")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Seed for the simulated reference designs")] = 0,
    n_sim: Annotated[Optional[int], typer.Option(
        "--n-sim", help="Simulated designs per candidate method (default depends on n)")] = None,
) -> None:
    """Infer how the input design was sampled (Grid, LHS, Sobol, Random).

    Compares stratification, discrepancy and nearest-neighbour spacing with
    designs simulated by each method. Only pass input columns: responses are
    not part of the design and make every design look like none of them.
    """
    df = pd.read_csv(file_path)
    if columns:
        missing = [c for c in _split(columns) if c not in df.columns]
        if missing:
            typer.echo(f"Columns not found: {', '.join(missing)}")
            raise typer.Exit(code=1)
        df = df[_split(columns)]
    df = df.drop(columns=[c for c in _split(exclude) if c in df.columns])
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        typer.echo("No numeric columns found in dataset")
        raise typer.Exit(code=1)
    try:
        result = detect_sampling_method(numeric_df.to_numpy(), n_sim=n_sim, random_state=seed)
    except ValueError as exc:
        typer.echo(f"Cannot test sampling method: {exc}")
        raise typer.Exit(code=1)
    typer.echo(f"Columns tested: {', '.join(numeric_df.columns)}")
    typer.echo(f"Detected sampling: {result.method}")
    typer.echo(f"Justification: {result.explanation}")
    if result.p_values:
        typer.echo("Monte Carlo p-values: " + ", ".join(f"{m}={p:.3f}" for m, p in result.p_values.items()))
    if not columns and not exclude:
        typer.echo("Note: all numeric columns were tested; use --exclude to drop response columns.")
