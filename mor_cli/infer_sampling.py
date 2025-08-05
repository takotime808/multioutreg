# Copyright (c) 2025 takotime808

import typer
import pandas as pd
from multioutreg.sampling.infer_sampling import infer_sampling_and_plot_umap

app = typer.Typer(name="multioutreg", rich_markup_mode="rich")

@app.command("infer_sampling")
def infer_sampling(csv_path: str):
    """Infer sampling method of the dataset in CSV."""
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include="number").values
    method, _, explanation = infer_sampling_and_plot_umap(X, explanation_indicator=True)
    typer.echo(f"Inferred Method: {method}")
    typer.echo(f"Explanation: {explanation}")