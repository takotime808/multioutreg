# Copyright (c) 2025 takotime808

import os
import typer
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from typing import Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

from multioutreg.model_selection import AutoDetectMultiOutputRegressor
from multioutreg.model_selection.evaluation import plan_evaluation, run_evaluation
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.utils.pca_utils import fit_input_pca
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test

from multioutreg.report.auto_detect_report import generate_html_report

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
    conformal: bool = typer.Option(False, help="Compute conformal prediction intervals"),
    conformal_alpha: float = typer.Option(0.1, help="Conformal miscoverage level (e.g. 0.1 for 90%% intervals)"),
    use_moe: bool = typer.Option(False, help="Include MoE (Mixture of Experts) as a joint multi-output candidate"),
    moe_n_experts: int = typer.Option(4, help="Number of MoE experts"),
    moe_gating_type: str = typer.Option("linear", help="MoE gating type: linear or mlp"),
    use_imputation: bool = typer.Option(False, help="Apply KNN imputation to fill missing values in selected columns before training"),
    description: str = typer.Option("", help="Project description"),
    out_html: str = typer.Option("model_report_auto.html", help="Output HTML file"),
) -> None:
    """Run autodetection and output an HTML report."""
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

    # How to estimate performance and calibrate intervals depends on n:
    # nested CV when a 25% test split would be tiny, CV+ when a 20%
    # calibration split is too small for conformal_alpha (AUDIT S1, S2).
    plan = plan_evaluation(len(X), alpha=conformal_alpha if conformal else None)
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

    def make_model():
        model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
        if use_moe:
            from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
            model.register_multi_output_candidates([
                MixtureOfExpertsSurrogate(
                    n_experts=moe_n_experts,
                    gating_type=moe_gating_type,
                    random_state=0,
                )
            ])
        return model

    result = run_evaluation(make_model, X, y, plan, preprocess, on_step=typer.echo)
    y_test, best_pred, best_std = result.y_true, result.pred, result.std
    X_train = result.X_fit
    best_model = result.model

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

    # Conformal prediction intervals
    conformal_summary_df = None
    conformal_intervals = None
    if conformal:
        y_lower, y_upper = result.lower, result.upper
        conformal_intervals = (y_lower, y_upper)
        from multioutreg.conformal.metrics import conformal_summary as _conformal_summary
        conformal_summary_df = _conformal_summary(
            y_test, y_lower, y_upper, alpha=conformal_alpha, output_names=out_cols
        )
        typer.echo(f"\nConformal Prediction Summary (alpha={conformal_alpha}):")
        typer.echo(conformal_summary_df.to_string(index=False))

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
        if conformal and conformal_intervals is not None:
            row = conformal_summary_df[conformal_summary_df["output"] == name]
            if not row.empty:
                metrics[name]["conformal_coverage"] = float(row["coverage"].iloc[0])
                metrics[name]["conformal_interval_width"] = float(row["mean_width"].iloc[0])

    html = generate_html_report(
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
        n_train=result.n_train,
        n_test=result.n_test,
        cross_validation=(f"Model selection: {best_model.cv_description_}. "
                          f"Performance estimate: {plan.describe()}"),
        seed=0,
        notes="Generated report.",
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
        pca_method=pca_method,
        pca_threshold=pca_threshold,
        pca_n_components=pca_n_components,
        kaiser_rule_suggestion=kaiser_rule_suggestion,
        conformal_intervals=conformal_intervals if conformal else None,
        conformal_alpha=conformal_alpha if conformal else None,
        imputation_summary=_imputation_summary,
    )

    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    typer.echo(f"Report written to {out_html}")


if __name__ == "__main__":
    app()