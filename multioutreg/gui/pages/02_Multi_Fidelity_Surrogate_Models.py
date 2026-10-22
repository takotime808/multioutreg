# Copyright (c) 2026 takotime808

"""Streamlit page for multi-fidelity surrogate modeling.

Accepts a single CSV with a fidelity level column, splits data per level,
and grid-searches over combinations of multi-fidelity surrogate types and
base surrogate classes:

* StackedVFMSurrogate  — nonlinear recursive feature augmentation (N levels)
* AdditiveCorrectionVFM — two-level additive correction (Kennedy-O'Hagan)
* MultiFidelitySurrogate — independent surrogate per level (no coupling)

Example data can be downloaded from the "Other Dataset Examples" page.
"""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from jinja2 import Template
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals_ordered,
    plot_conformal_vs_gaussian,
)
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
from multioutreg.figures.shap_multioutput import generate_shap_plot
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.surrogates import (
    AdditiveCorrectionVFM,
    GaussianProcessSurrogate,
    HistGradientBoostingSurrogate,
    LinearRegressionSurrogate,
    MultiFidelitySurrogate,
    RandomForestSurrogate,
    StackedVFMSurrogate,
    ElasticNetSurrogate,
    LassoSurrogate,
)
from multioutreg.surrogates.lightgbm_sklearn import _LIGHTGBM_AVAILABLE, LightGBMSurrogate
from multioutreg.surrogates.xgboost_sklearn import _XGBOOST_AVAILABLE, XGBoostSurrogate
from multioutreg.surrogates.catboost_sklearn import _CATBOOST_AVAILABLE, CatBoostSurrogate
from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE, SparseGPSurrogate
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.utils.imputation import apply_imputation, detect_missing


# ---------------------------------------------------------------------------
# HTML report helper (mirrors Grid_Search_Surrogate_Models.py)
# ---------------------------------------------------------------------------

def generate_html_report(
    model_type: str,
    fidelity_levels: List[str],
    output_names: List[str],
    description: str,
    metrics: Dict[str, Dict[str, float]],
    uncertainty_metrics: Dict[str, float],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    best_std: np.ndarray,
    best_model: Any,
    X_train: np.ndarray,
    n_train: int,
    n_test: int,
    cross_validation: str,
    seed: int,
    notes: str,
    conformal_intervals: tuple | None = None,
    conformal_alpha: float | None = None,
    imputation_summary: dict | None = None,
) -> str:
    """Render a Jinja2 HTML report from model evaluation results."""
    prediction_plots: Dict[str, str] = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi,
        best_pred,
        best_std,
        y_test,
        target_list=output_names,
    )
    for i, name in enumerate(output_names):
        prediction_plots[name] = safe_plot_b64(
            plot_predictions_with_error_bars,
            y_test[:, [i]],
            best_pred[:, [i]],
            best_std[:, [i]],
            output_names=[name],
            n_cols=3,
        )

    shap_plots = generate_shap_plot(best_model, X_train, output_names)

    unc_img = safe_plot_b64(
        plot_coverage, y_test, best_pred, best_std, output_names=output_names
    )
    uncertainty_plots = [
        {
            "img_b64": unc_img,
            "title": "Coverage Plot",
            "caption": "Nominal vs empirical coverage.",
        }
    ]

    pdp_plots = generate_pdp_plot(
        best_model, X_train, output_names, feature_names=list(output_names)
    )

    sampling_umap_plot, sampling_method_explanation = generate_umap_plot(X_train)

    other_img = safe_plot_b64(
        plot_residuals_multioutput_with_regplot,
        best_pred,
        y_test,
        target_list=output_names,
    )
    sampling_other_plots = [
        {
            "img_b64": other_img,
            "title": "Residuals",
            "caption": "Residual vs predicted values.",
        }
    ]

    other_plots = generate_error_histogram(y_test, best_pred, output_names)

    conformal_plots_list = []
    if conformal_intervals is not None and conformal_alpha is not None:
        y_lower, y_upper = conformal_intervals
        alpha = conformal_alpha

        conformal_plots_list.append({
            "img_b64": safe_plot_b64(
                plot_conformal_intervals_ordered,
                y_test, y_lower, y_upper,
                y_pred=best_pred,
                output_names=output_names,
                alpha=alpha,
            ),
            "title": f"Conformal Prediction Intervals ({int((1 - alpha) * 100)}%)",
            "caption": (
                "Distribution-free prediction intervals ordered by observed value. "
                "Guaranteed marginal coverage."
            ),
        })
        conformal_plots_list.append({
            "img_b64": safe_plot_b64(
                plot_conformal_vs_gaussian,
                y_test, best_pred, best_std,
                np.abs(y_test - best_pred),
                output_names=output_names,
            ),
            "title": "Gaussian CI vs Conformal PI Coverage",
            "caption": (
                "Comparison of nominal vs empirical coverage for Gaussian confidence "
                "intervals and conformal prediction intervals."
            ),
        })

    template_path = os.path.join(
        os.path.dirname(__file__), "../../report/template.html"
    )
    env_override = os.environ.get("MOR_TEMPLATE_PATH")
    if env_override:
        template_path = env_override
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    rendered = Template(template_text).render(
        project_title="Multi-Fidelity Surrogate Modeling Report",
        model_type=model_type,
        fidelity_levels=fidelity_levels,
        output_names=output_names,
        description=description,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
        uncertainty_plots=uncertainty_plots,
        prediction_plots=prediction_plots,
        shap_plots=shap_plots,
        pdp_plots=pdp_plots,
        sampling_umap_plot=sampling_umap_plot,
        sampling_method_explanation=sampling_method_explanation,
        sampling_other_plots=sampling_other_plots,
        other_plots=other_plots,
        n_train=n_train,
        n_test=n_test,
        cross_validation=cross_validation,
        seed=seed,
        notes=notes,
        feature_names_pca=None,
        pca_explained_variance=None,
        pca_variance_plot=None,
        pca_method=None,
        pca_threshold=None,
        pca_n_components=None,
        kaiser_rule_suggestion=None,
        conformal_plots=conformal_plots_list,
        imputation_summary=imputation_summary,
    )
    return rendered


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _split_by_fidelity(
    df: pd.DataFrame,
    fidelity_col: str,
    input_cols: List[str],
    output_cols: List[str],
    ordered_levels: List,
) -> Dict[str, tuple]:
    """Return ``{level_name: (X, Y)}`` for each fidelity level."""
    level_data: Dict[str, tuple] = {}
    for raw in ordered_levels:
        mask = df[fidelity_col] == raw
        sub = df[mask]
        X_k = sub[input_cols].values.astype(np.float64)
        Y_k = sub[output_cols].values.astype(np.float64)
        level_data[str(raw)] = (X_k, Y_k)
    return level_data


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    output_names: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(output_names):
        metrics[name] = {
            "r2": r2_score(y_test[:, i], y_pred[:, i]),
            "rmse": root_mean_squared_error(y_test[:, i], y_pred[:, i]),
            "mae": mean_absolute_error(y_test[:, i], y_pred[:, i]),
            "mean_predicted_std": float(np.mean(y_std[:, i])),
        }
    return metrics


# ---------------------------------------------------------------------------
# Grid search helpers
# ---------------------------------------------------------------------------

def _build_candidates(level_names: List[str], skip_expensive: bool) -> List[dict]:
    """Return a list of candidate configs for the grid search.

    Each config is a dict with keys:
        name            display name for the results table
        mf_type         "stacked" | "additive" | "independent"
        surrogate_cls   base surrogate class
        augment_with_std  bool (StackedVFM only)
    """
    surrogates: List[tuple] = [
        ("RF", RandomForestSurrogate),
        ("HGB", HistGradientBoostingSurrogate),
        ("Linear", LinearRegressionSurrogate),
        ("ElasticNet", ElasticNetSurrogate),
        ("Lasso", LassoSurrogate),
    ]
    if not skip_expensive:
        surrogates.append(("GP", GaussianProcessSurrogate))
    if _LIGHTGBM_AVAILABLE:
        surrogates.append(("LightGBM", LightGBMSurrogate))
    if _XGBOOST_AVAILABLE:
        surrogates.append(("XGBoost", XGBoostSurrogate))
    if _CATBOOST_AVAILABLE:
        surrogates.append(("CatBoost", CatBoostSurrogate))
    if _GPYTORCH_AVAILABLE:
        surrogates.append(("SparseGP", SparseGPSurrogate))

    candidates: List[dict] = []
    for sur_name, sur_cls in surrogates:
        # StackedVFM — without and with std augmentation
        candidates.append({
            "name": f"StackedVFM+{sur_name}",
            "mf_type": "stacked",
            "surrogate_cls": sur_cls,
            "augment_with_std": False,
        })
        candidates.append({
            "name": f"StackedVFM+{sur_name}+augStd",
            "mf_type": "stacked",
            "surrogate_cls": sur_cls,
            "augment_with_std": True,
        })
        # MultiFidelitySurrogate (independent per level, no coupling)
        candidates.append({
            "name": f"MultiFidelity+{sur_name}",
            "mf_type": "independent",
            "surrogate_cls": sur_cls,
            "augment_with_std": False,
        })
        # AdditiveCorrectionVFM — 2-level only
        if len(level_names) == 2:
            candidates.append({
                "name": f"AdditiveVFM+{sur_name}",
                "mf_type": "additive",
                "surrogate_cls": sur_cls,
                "augment_with_std": False,
            })

    return candidates


def _fit_candidate(
    cand: dict,
    train_data: Dict[str, tuple],
    level_names: List[str],
) -> Any:
    """Fit and return the model described by *cand*."""
    sur_cls = cand["surrogate_cls"]

    if cand["mf_type"] == "stacked":
        model = StackedVFMSurrogate(
            fidelity_levels=level_names,
            surrogate_cls=sur_cls,
            augment_with_std=cand["augment_with_std"],
        )
        model.fit(train_data)

    elif cand["mf_type"] == "additive":
        lo_name, hi_name = level_names[0], level_names[1]
        model = AdditiveCorrectionVFM(
            lo_surrogate_cls=sur_cls,
            hi_surrogate_cls=sur_cls,
        )
        model.fit({"lo": train_data[lo_name], "hi": train_data[hi_name]})

    else:  # independent
        model = MultiFidelitySurrogate(sur_cls, level_names)
        model.fit(train_data)

    return model


def _predict_highest(
    model: Any,
    cand: dict,
    X: np.ndarray,
    level_names: List[str],
) -> tuple:
    """Return highest-fidelity ``(mean, std)`` predictions as 2-D float arrays."""
    if cand["mf_type"] == "independent":
        # MultiFidelitySurrogate.predict defaults to the *lowest* level.
        y_pred, y_std = model.predict(X, level=level_names[-1], return_std=True)
    else:
        y_pred, y_std = model.predict(X, return_std=True)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_std.ndim == 1:
        y_std = y_std.reshape(-1, 1)
    return y_pred, y_std


def _select_best_candidate(
    candidates: List[dict],
    train_data: Dict[str, tuple],
    level_names: List[str],
    val_size: float = 0.2,
    random_state: int = 0,
    on_progress: Any = None,
) -> tuple:
    """Grid-search *candidates* on a validation split of the highest-fidelity
    training data and return ``(results, best_cand)``.

    The test set is never seen here, so the winner's test-set metrics are an
    honest estimate rather than the best of N scores on the same data.
    """
    hi_name = level_names[-1]
    X_hi, Y_hi = train_data[hi_name]
    if len(X_hi) < 5:
        raise ValueError(
            f"Highest-fidelity level '{hi_name}' has only {len(X_hi)} training "
            "samples; at least 5 are needed to hold out a validation split."
        )
    X_fit, X_val, Y_fit, Y_val = train_test_split(
        X_hi, Y_hi, test_size=val_size, random_state=random_state
    )
    fit_data = {**train_data, hi_name: (X_fit, Y_fit)}

    results: List[dict] = []
    best_cand = None
    best_mse = np.inf
    for i, cand in enumerate(candidates):
        if on_progress is not None:
            on_progress(i, cand)
        try:
            model = _fit_candidate(cand, fit_data, level_names)
            y_pred, _ = _predict_highest(model, cand, X_val, level_names)
            mse = float(np.mean((Y_val - y_pred) ** 2))
            results.append({
                "Model": cand["name"],
                "Val MSE": round(mse, 6),
                "Val RMSE": round(float(np.sqrt(mse)), 6),
                "Val R²": round(float(r2_score(Y_val.ravel(), y_pred.ravel())), 4),
                "Status": "OK",
            })
            if mse < best_mse:
                best_mse = mse
                best_cand = cand
        except Exception as exc:
            results.append({
                "Model": cand["name"],
                "Val MSE": None,
                "Val RMSE": None,
                "Val R²": None,
                "Status": f"Error: {exc}",
            })
    return results, best_cand


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------

st.title("Multi-Fidelity Surrogate Model")
st.info(
    "Upload a single CSV that contains a **fidelity level column** alongside your input "
    "features and output targets.  Each unique value in that column defines one fidelity "
    "level (e.g. 0 = low, 1 = medium, 2 = high).  \n\n"
    "The page will **grid-search** over all combinations of multi-fidelity model type "
    "and base surrogate class, scoring each on a validation split of the highest-fidelity "
    "training data. The best configuration is refit on all training data, evaluated once "
    "on the held-out test set, and reported.\n\n"
    "Example data can be downloaded from the **Other Dataset Examples** page."
)

uploaded_file = st.file_uploader(
    "Upload CSV file with a fidelity level column.",
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview", df.head())

    # ------------------------------------------------------------------
    # Missing value handling
    # ------------------------------------------------------------------
    _missing_summary = detect_missing(df)
    _impute_choices: Dict[str, str] = {}
    if not _missing_summary.empty:
        st.warning(
            f"{int(_missing_summary['missing_count'].sum())} missing value(s) detected "
            f"across {len(_missing_summary)} column(s)."
        )
        st.dataframe(_missing_summary)
        st.write("**Choose how to handle missing values per column:**")
        for _col in _missing_summary.index:
            _impute_choices[_col] = st.selectbox(
                f"`{_col}`",
                ["Impute (KNN)", "Drop rows"],
                key=f"imp_{_col}",
            )

    # ------------------------------------------------------------------
    # Fidelity column selector — outside the form so changing it
    # immediately recomputes `remaining` and the level options.
    # ------------------------------------------------------------------
    st.subheader("Column Selection")
    fidelity_col = st.selectbox(
        "Fidelity level column",
        options=df.columns,
        help="Column whose unique values identify fidelity levels (e.g. 0, 1, 2 or 'lo', 'hi').",
        key="fidelity_col_select",
    )
    remaining = [c for c in df.columns if c != fidelity_col]
    _detected = sorted(df[fidelity_col].dropna().unique().tolist(), key=str)

    # ------------------------------------------------------------------
    # Column selection + grid search options form
    # ------------------------------------------------------------------
    with st.form("mf_column_selection"):
        input_cols = st.multiselect("Input feature columns", options=remaining)
        output_cols = st.multiselect("Output target columns", options=remaining)

        st.subheader("Fidelity Level Ordering")
        ordered_levels = st.multiselect(
            "Fidelity levels — ordered lowest → highest",
            options=_detected,
            default=_detected,
            help=(
                "Re-order by removing and re-adding levels in the correct order. "
                "The last entry is treated as the highest fidelity."
            ),
        )

        st.subheader("Grid Search Options")
        skip_expensive = st.checkbox(
            "Skip computationally expensive models (Gaussian Process)",
            value=True,
            help=(
                "Gaussian Process has O(n³) training cost and becomes very slow for "
                "n > ~300.  Uncheck to include GP-based candidates in the grid search."
            ),
        )
        use_conformal = st.checkbox(
            "Compute conformal prediction intervals",
            value=False,
            help="Calibrate split-conformal intervals on a held-out part of the highest-fidelity training data.",
        )
        conformal_alpha_sel = st.slider(
            "Conformal alpha (miscoverage level)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Target miscoverage rate α; intervals achieve ≥ 1−α marginal coverage.",
        )

        description = st.text_area("Optional: project description")
        submitted = st.form_submit_button("Run Grid Search")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if submitted:
        if not input_cols:
            st.error("Please select at least one input feature column.")
            st.stop()
        if not output_cols:
            st.error("Please select at least one output target column.")
            st.stop()
        if len(ordered_levels) < 2:
            st.error("Please select at least 2 fidelity levels.")
            st.stop()

        # Apply imputation
        _all_selected = list(input_cols) + list(output_cols)
        _cols_to_impute = [
            c for c in _all_selected
            if st.session_state.get(f"imp_{c}") == "Impute (KNN)"
        ]
        _cols_to_drop = [
            c for c in _all_selected
            if st.session_state.get(f"imp_{c}") == "Drop rows"
        ]
        _rows_before = len(df)
        if _cols_to_impute or _cols_to_drop:
            df = apply_imputation(df, _cols_to_impute, _cols_to_drop)
        _rows_after = len(df)

        _imputation_summary = None
        if (_cols_to_impute or _cols_to_drop) and not _missing_summary.empty:
            _summary_cols = []
            for _c in _cols_to_impute:
                if _c in _missing_summary.index:
                    _summary_cols.append({
                        "name": _c,
                        "action": "Imputed (KNN)",
                        "missing_count": int(_missing_summary.loc[_c, "missing_count"]),
                        "missing_pct": float(_missing_summary.loc[_c, "missing_pct"]),
                    })
            for _c in _cols_to_drop:
                if _c in _missing_summary.index:
                    _summary_cols.append({
                        "name": _c,
                        "action": "Rows dropped",
                        "missing_count": int(_missing_summary.loc[_c, "missing_count"]),
                        "missing_pct": float(_missing_summary.loc[_c, "missing_pct"]),
                    })
            _imputation_summary = {
                "rows_before": _rows_before,
                "rows_after": _rows_after,
                "columns": _summary_cols,
            }

        # Split by fidelity level
        level_names = [str(lv) for lv in ordered_levels]
        level_data = _split_by_fidelity(
            df, fidelity_col, list(input_cols), list(output_cols), ordered_levels
        )

        # Warn on small levels
        for lv_name, (X_k, _) in level_data.items():
            if X_k.shape[0] < 10:
                st.warning(
                    f"Level '{lv_name}' has only {X_k.shape[0]} samples — "
                    "results may be unreliable."
                )

        # Train/test split per level (80/20); evaluate on highest-fidelity test set
        train_data: Dict[str, tuple] = {}
        test_data: Dict[str, tuple] = {}
        for lv_name, (X_k, Y_k) in level_data.items():
            X_tr, X_te, Y_tr, Y_te = train_test_split(
                X_k, Y_k, test_size=0.25, random_state=0
            )
            train_data[lv_name] = (X_tr, Y_tr)
            test_data[lv_name] = (X_te, Y_te)

        X_test, y_test = test_data[level_names[-1]]

        # Conformal calibration must use data disjoint from both training and the
        # test set used to report coverage, so hold out part of the hi-fi training split.
        if use_conformal:
            _X_hi, _Y_hi = train_data[level_names[-1]]
            _X_fit, X_cal, _Y_fit, y_cal = train_test_split(
                _X_hi, _Y_hi, test_size=0.2, random_state=0
            )
            train_data[level_names[-1]] = (_X_fit, _Y_fit)

        X_train_hi, _ = train_data[level_names[-1]]  # used for SHAP / plots
        output_names = list(output_cols)

        # ------------------------------------------------------------------
        # Grid search
        # ------------------------------------------------------------------
        candidates = _build_candidates(level_names, skip_expensive)

        prog = st.progress(0, text="Running grid search…")

        def _on_progress(_i: int, _cand: dict) -> None:
            prog.progress(
                (_i + 1) / len(candidates),
                text=f"Fitting {_cand['name']} ({_i + 1}/{len(candidates)})…",
            )

        try:
            gs_results, best_cand = _select_best_candidate(
                candidates, train_data, level_names, on_progress=_on_progress
            )
        except ValueError as exc:
            prog.empty()
            st.error(str(exc))
            st.stop()

        prog.empty()

        if best_cand is None:
            st.error(
                "All model candidates failed. "
                "Check the error messages in the results table below."
            )
            st.dataframe(pd.DataFrame(gs_results))
            st.stop()

        # Refit the winner on all training data; the test set is used only here.
        best_name = best_cand["name"]
        best_model = _fit_candidate(best_cand, train_data, level_names)
        best_pred, best_std = _predict_highest(best_model, best_cand, X_test, level_names)

        st.write(f"### Grid Search Results (validation split) — best: **{best_name}**")
        _gs_df = (
            pd.DataFrame(gs_results)
            .sort_values("Val MSE", na_position="last")
            .reset_index(drop=True)
        )
        st.dataframe(_gs_df, width="stretch")

        # Per-output metrics for the best model
        metrics = _compute_metrics(y_test, best_pred, best_std, output_names)
        st.write("### Best Model Metrics (highest-fidelity test set)")
        st.dataframe(pd.DataFrame(metrics).T)

        # Conformal prediction
        conformal_intervals = None
        if use_conformal:
            from multioutreg.conformal.base import BaseConformalPredictor
            from multioutreg.conformal.metrics import conformal_summary as _conf_summary

            _cal_pred, _ = _predict_highest(best_model, best_cand, X_cal, level_names)
            residuals = np.abs(y_cal - _cal_pred)
            n_outputs = y_test.shape[1]
            q = np.array([
                BaseConformalPredictor._conformal_quantile(
                    residuals[:, j], conformal_alpha_sel
                )
                for j in range(n_outputs)
            ])
            y_lower = best_pred - q[np.newaxis, :]
            y_upper = best_pred + q[np.newaxis, :]
            conformal_intervals = (y_lower, y_upper)

            conf_df = _conf_summary(
                y_test, y_lower, y_upper,
                alpha=conformal_alpha_sel,
                output_names=output_names,
            )
            st.write(f"### Conformal Prediction Summary (alpha={conformal_alpha_sel})")
            st.dataframe(conf_df)

        # Report
        try:
            with st.spinner("Generating HTML report…"):
                html = generate_html_report(
                    model_type=best_model.__class__.__name__,
                    fidelity_levels=level_names,
                    output_names=output_names,
                    description=description,
                    metrics=metrics,
                    uncertainty_metrics={"dummy_metric": 0.0},
                    y_test=y_test,
                    best_pred=best_pred,
                    best_std=best_std,
                    best_model=best_model,
                    X_train=X_train_hi,
                    n_train=X_train_hi.shape[0],
                    n_test=X_test.shape[0],
                    cross_validation="Hold-out validation (20% of highest-fidelity training data)",
                    seed=0,
                    notes=description or "Generated by Multi-Fidelity Surrogate Model page.",
                    conformal_intervals=conformal_intervals if use_conformal else None,
                    conformal_alpha=conformal_alpha_sel if use_conformal else None,
                    imputation_summary=_imputation_summary,
                )
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")
            st.stop()

        st.download_button(
            "Download HTML Report",
            html,
            file_name="mf_model_report.html",
            mime="text/html",
        )
