# Copyright (c) 2025 takotime808

import typer
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from typing import Optional
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.utils.pca_utils import fit_input_pca
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test
from multioutreg.gui.grid_search_core import (
    PerTargetGridSearch,
    generate_html_report,
)
from multioutreg.model_selection.evaluation import plan_evaluation, run_evaluation

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
    use_moe: bool = typer.Option(False, help="Include MoE (Mixture of Experts) as a joint multi-output candidate"),
    moe_n_experts: int = typer.Option(4, help="Number of MoE experts"),
    moe_gating_type: str = typer.Option("linear", help="MoE gating type: linear or mlp"),
    use_imputation: bool = typer.Option(False, help="Apply KNN imputation to fill missing values in selected columns before training"),
    description: str = typer.Option("", help="Project description"),
    out_html: str = typer.Option("model_report.html", help="Output HTML file"),
) -> None:
    """Run the grid search and output an HTML report."""
    df = pd.read_csv(data_path)
    in_cols = [c.strip() for c in input_cols.split(',')]
    out_cols = [c.strip() for c in output_cols.split(',')]
    _imputation_summary = None
    if use_imputation:
        _impute_cols = in_cols + out_cols
        _before_missing = detect_missing(df[_impute_cols])
        _rows_before = len(df)
        # Targets are never imputed; inputs are imputed after the split (below).
        df = drop_unusable_rows(df, out_cols, cols_to_drop_rows=[])
        _rows_after = len(df)
        _imputation_summary = {
            "rows_before": _rows_before,
            "rows_after": _rows_after,
            "columns": [
                {
                    "name": c,
                    "action": (
                        "Rows dropped (missing target)" if c in out_cols
                        else "Imputed (KNN, fit on training split)"
                    ),
                    "missing_count": int(_before_missing.loc[c, "missing_count"]),
                    "missing_pct": float(_before_missing.loc[c, "missing_pct"]),
                }
                for c in _before_missing.index
            ],
        }
    X = df[in_cols].values
    y = df[out_cols].values

    # Nested CV when a 25% test split would be tiny (AUDIT S1).
    plan = plan_evaluation(len(X))
    for note in plan.notes:
        typer.echo(note)

    def preprocess(X_tr, X_te):
        # Imputation and PCA are fitted on the training rows only.
        if use_imputation:
            X_tr, X_te = impute_train_test(X_tr, X_te)
        pca = None
        if use_pca:
            pca = fit_input_pca(X_tr, X_te, pca_method, n_components, pca_threshold)
            X_tr, X_te = pca.X_train, pca.X_test
        return X_tr, X_te, pca

    moe_factory = None
    if use_moe:
        from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate

        def moe_factory():
            return MixtureOfExpertsSurrogate(
                n_experts=moe_n_experts, gating_type=moe_gating_type, random_state=0)

    # Configs are selected by CV on the training rows; the evaluation rows
    # (holdout, or the outer folds of nested CV) are only used for the metrics below.
    result = run_evaluation(
        lambda: PerTargetGridSearch(moe_factory=moe_factory), X, y, plan, preprocess,
        on_step=typer.echo)
    y_test, best_pred, best_std = result.y_true, result.pred, result.std
    X_train = result.X_fit
    search = result.model.result_
    best_model = search.model
    if search.moe_selected:
        typer.echo("MoE outperformed all per-target models and was selected.")

    pca_variance_plot = None
    pca_explained_variance = None
    pca_n_components = None
    feature_names_pca = None
    kaiser_rule_suggestion = None

    if use_pca:
        _pca = result.info
        pca_n_components = _pca.n_components
        feature_names_pca = _pca.feature_names
        pca_explained_variance = _pca.preview_pca.explained_variance_ratio_.tolist()
        pca_variance_plot = generate_pca_variance_plot(
            _pca.preview_pca,
            n_selected=pca_n_components,
            threshold=pca_threshold if pca_method == "Explained variance threshold" else None,
        )
        kaiser_rule_suggestion = (
            f"Kaiser rule suggests **{_pca.kaiser_k}** components "
            "(eigenvalues > 1 on standardized inputs)."
        )

    metrics = {}
    for i, name in enumerate(out_cols):
        y_true = y_test[:, i]
        y_pred = best_pred[:, i]
        metrics[name] = {
            "r2": r2_score(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
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
        n_train=result.n_train,
        n_test=result.n_test,
        cross_validation=(
            f"Model selected per output by {search.cv_description} on the training data. "
            f"Performance estimate: {plan.describe()}"
        ),
        seed=0,
        notes="Generated report.",
        feature_names=feature_names_pca if use_pca else in_cols,
        feature_names_pca=feature_names_pca,
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
        pca_method=pca_method,
        pca_threshold=pca_threshold,
        pca_n_components=pca_n_components,
        kaiser_rule_suggestion=kaiser_rule_suggestion,
        imputation_summary=_imputation_summary,
    )

    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    typer.echo(f"Report written to {out_html}")


if __name__ == "__main__":
    app()