# Copyright (c) 2025 takotime808

import typer
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from typing import Optional
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA

from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.gui.Grid_Search_Surrogate_Models import (
    RandomForestWithUncertainty,
    GradientBoostingWithUncertainty,
    KNeighborsRegressorWithUncertainty,
    BootstrapLinearRegression,
    PerTargetRegressorWithStd,
    generate_html_report,
)

app = typer.Typer(help="Grid search surrogate models and generate a report")


@app.command()
def grid_search(
    data_path: str = typer.Argument(..., help="Path to CSV data"),
    input_cols: str = typer.Argument(..., help="Comma separated input columns"),
    output_cols: str = typer.Argument(..., help="Comma separated output columns"),
    use_pca: bool = typer.Option(False, help="Apply PCA to inputs"),
    pca_method: Optional[str] = typer.Option(None, help="PCA selection method: Manual, Explained variance threshold, Kaiser rule"),
    n_components: Optional[int] = typer.Option(None, help="Number of PCA components when manual"),
    pca_threshold: Optional[float] = typer.Option(None, help="Explained variance threshold"),
    description: str = typer.Option("", help="Project description"),
    out_html: str = typer.Option("model_report.html", help="Output HTML file"),
) -> None:
    """Run the grid search and output an HTML report."""
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

    surrogate_defs = [
        ("gpr", GaussianProcessRegressor, {"alpha": [1e-4], "kernel": [RBF(), Matern(nu=1.5)]}),
        ("rf", RandomForestWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
        ("gb", GradientBoostingWithUncertainty, {"alpha": [0.95], "n_estimators": [50]}),
        ("knn", KNeighborsRegressorWithUncertainty, {"n_neighbors": [3]}),
        ("blr", BootstrapLinearRegression, {"n_bootstraps": [20]}),
    ]

    configs = [(name, Est, params) for name, Est, grid in surrogate_defs for params in ParameterGrid(grid)]

    best_score = float("inf")
    best_combo = None
    best_pred = None
    best_std = None
    best_model = None

    for est_params in ParameterGrid({"combos": [configs] * y_train.shape[1]}):
        try:
            combo = est_params["combos"][:y_train.shape[1]]
            estimators = [est(**params) for (_, est, params) in combo]
            model = PerTargetRegressorWithStd(estimators)
            model.fit(X_train, y_train)
            pred, std = model.predict(X_test, return_std=True)
            score = mean_squared_error(y_test, pred)
            if score < best_score:
                best_score = score
                best_combo = [dict(model=name, **params) for (name, _, params) in combo]
                best_pred, best_std = pred, std
                best_model = model
        except Exception:
            continue

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

    html = generate_html_report(
        model_type="PerTargetRegressorWithStd",
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