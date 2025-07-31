# Copyright (c) 2025 takotime808

import os
import typer
import pandas as pd
import numpy as np
import importlib.util
import matplotlib
matplotlib.use('Agg')
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from multioutreg.model_selection import AutoDetectMultiOutputRegressor
from multioutreg.figures.pca_plots import generate_pca_variance_plot

# # NOTE: Module starts with numeric values, so it cannot be directly imported
# from multioutreg.gui.pages.Use_Auto_Detect_MultiOutput_Regressor import generate_html_report
file_path = os.path.abspath("multioutreg/gui/pages/01_Use_Auto_Detect_MultiOutput_Regressor.py")
spec = importlib.util.spec_from_file_location("use_auto_detect", file_path)
use_auto_detect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(use_auto_detect)

app = typer.Typer(help="Auto-detect surrogate models and generate a report")


@app.command()
def grid_search_auto_detect(
    data_path: str = typer.Argument(..., help="Path to CSV data"),
    input_cols: str = typer.Argument(..., help="Comma separated input columns"),
    output_cols: str = typer.Argument(..., help="Comma separated output columns"),
    use_pca: bool = typer.Option(False, help="Apply PCA to inputs"),
    pca_method: Optional[str] = typer.Option(None, help="PCA selection method"),
    n_components: Optional[int] = typer.Option(None, help="Number of PCA components"),
    pca_threshold: Optional[float] = typer.Option(None, help="Explained variance threshold"),
    description: str = typer.Option("", help="Project description"),
    out_html: str = typer.Option("model_report_auto.html", help="Output HTML file"),
) -> None:
    """Run autodetection and output an HTML report."""
    df = pd.read_csv(data_path)
    in_cols = [c.strip() for c in input_cols.split(',')]
    out_cols = [c.strip() for c in output_cols.split(',')]
    X = df[in_cols].values
    y = df[out_cols].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    pca_variance_plot = None
    pca_explained_variance = None
    pca_n_components = None
    feature_names_pca = None
    kaiser_rule_suggestion = None

    if use_pca:
        preview_pca = PCA().fit(X_train)
        kaiser_k = int(np.sum(preview_pca.explained_variance_ > 1))
        if pca_method == "Manual" and n_components is not None:
            pca_n_components = int(n_components)
        elif pca_method == "Explained variance threshold" and pca_threshold is not None:
            cum = np.cumsum(preview_pca.explained_variance_ratio_)
            pca_n_components = int(np.searchsorted(cum, pca_threshold) + 1)
        else:
            pca_n_components = max(1, kaiser_k)
        pca = PCA(n_components=pca_n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        feature_names_pca = [f"PC{i+1}" for i in range(pca_n_components)]
        pca_explained_variance = preview_pca.explained_variance_ratio_.tolist()
        pca_variance_plot = generate_pca_variance_plot(
            preview_pca,
            n_selected=pca_n_components,
            threshold=pca_threshold if pca_method == "Explained variance threshold" else None,
        )
        kaiser_rule_suggestion = f"Kaiser rule suggests **{kaiser_k}** components (eigenvalues > 1)."

    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
    model.fit(X_train, y_train)
    best_pred, best_std = model.predict(X_test, return_std=True)
    best_model = model

    metrics = {}
    for i, name in enumerate(out_cols):
        y_true = y_test[:, i]
        y_pred = best_pred[:, i]
        metrics[name] = {
            "r2": r2_score(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "mae": mean_absolute_error(y_true, y_pred),
            "mean_predicted_std": float(np.mean(best_std[:, i])),
        }

    html = use_auto_detect.generate_html_report(
        model_type="AutoDetectMultiOutputRegressor",
        fidelity_levels=[],
        output_names=out_cols,
        description=description,
        metrics=metrics,
        uncertainty_metrics={"dummy_metric": 0.0},
        y_test=y_test,
        best_pred=best_pred,
        best_std=best_std,
        best_model=best_model,
        X_train=X_train,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        cross_validation="None",
        seed=0,
        notes="Generated report.",
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
        pca_method=pca_method,
        pca_threshold=pca_threshold,
        pca_n_components=pca_n_components,
        kaiser_rule_suggestion=kaiser_rule_suggestion,
    )

    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    typer.echo(f"Report written to {out_html}")


if __name__ == "__main__":
    app()