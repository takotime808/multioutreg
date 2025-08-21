# Copyright (c) 2025 takotime808

import typer
import pandas as pd
from typing_extensions import Annotated
from multioutreg.figures.umap_plot_classify import generate_umap_plot

# Arg = typer.Argument
# Opt = typer.Option
app = typer.Typer(
    name="multioutreg",
    rich_markup_mode="rich",
)

@app.command(
    "detect_sampling",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def detect_sampling(
    file_path: Annotated[str, typer.Argument(..., help="Path to CSV dataset")]
) -> None:
    """Infer the dataset sampling method using UMAP heuristics."""
    df = pd.read_csv(file_path)
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        typer.echo("No numeric columns found in dataset")
        raise typer.Exit(code=1)
    _, explanation = generate_umap_plot(numeric_df.to_numpy())
    method = explanation.split("->")[-1].strip()
    typer.echo(f"Detected sampling: {method}")
    typer.echo(f"Justification: {explanation}")