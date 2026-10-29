# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import io
import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from jinja2 import Template
try:
    import torch as _torch
    _torch_available = True
except ImportError:
    _torch_available = False

# from multioutreg.gui.report_plotting_utils import (
#     # plot_to_b64,
#     generate_prediction_plot,
#     generate_shap_plot,
#     generate_pdp_plot,
#     generate_uncertainty_plots,
#     generate_umap_plot,
#     generate_error_histogram
#     generate_pca_variance_plot,
# )

from multioutreg.utils.thread_limits import limit_native_threads, max_threads
from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.shap_multioutput import (
    generate_shap_plot,
    plot_multioutput_shap_bar_subplots,
)
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.utils.pca_utils import fit_input_pca
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
# from multioutreg.figures.shap_multioutput import plot_multioutput_shap_bar_subplots
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
# from multioutreg.figures.prediction_plots import plot_predictions
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals_ordered,
    plot_conformal_vs_gaussian,
)
from multioutreg.model_selection import AutoDetectMultiOutputRegressor
from multioutreg.model_selection.auto_detect import EXPENSIVE_MODEL_NAMES, drop_candidates
from multioutreg.model_selection.evaluation import plan_evaluation, run_evaluation
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test
from multioutreg.utils.metric_utils import gaussian_uncertainty_metrics
from multioutreg.time_series.ts_suitability import check_ts_suitability
from multioutreg.metrics import hypervolume_regret, scalarized_regret, epsilon_regret

# # NOTE: NOT used...yet.
# from multioutreg.figures.doe_plots import make_doe_plot
# from multioutreg.figures.model_comparison import plot_surrogate_model_summary



from multioutreg.report.auto_detect_report import generate_html_report
from multioutreg.report.render import render_styles
from multioutreg.gui.report_controls import report_downloads_all, report_format_control


# ----- Regret / download helpers -----
def _serialize_model(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def _parse_optional_float(txt: Optional[str]) -> Optional[float]:
    if not txt or not txt.strip():
        return None
    try:
        return float(txt.strip())
    except ValueError:
        return None


def _parse_csv_floats(txt: str) -> List[float]:
    return [float(x) for x in re.split(r"[,\s]+", txt.strip()) if x]


def _parse_weight_matrix(txt: str, m: int) -> Optional[np.ndarray]:
    """One weight vector per line (comma/space separated). Returns (k, m) rows on the simplex, or None if invalid."""
    rows = []
    try:
        for line in (txt or "").strip().splitlines():
            if line.strip():
                vals = _parse_csv_floats(line)
                if len(vals) != m:
                    return None
                rows.append(vals)
    except ValueError:
        return None
    if not rows:
        return None
    W = np.clip(np.asarray(rows, dtype=float), 0.0, np.inf)
    sums = W.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return W / sums


def _auto_reference_point(Y_true: np.ndarray, margin_pct: float) -> np.ndarray:
    """Minimization reference point: worst true value + margin_pct% of span (1.0 fallback when span == 0).

    Decision regret only ever measures hypervolume of true values, so the predictions
    do not enter the reference point.
    """
    Y_true = np.asarray(Y_true, dtype=float)
    span = np.ptp(Y_true, axis=0)
    return Y_true.max(axis=0) + np.where(span > 0, (margin_pct / 100.0) * span, 1.0)


def _regret_status_line(label: str, value: Optional[float], bound: Optional[float]) -> str:
    if value is None:
        return f"{label}: (not computed)"
    if bound is None:
        return f"{label}: {value:.4g}"
    icon = "✅" if value <= bound else "❌"
    return f"{icon} {label}: {value:.4g} (bound ≤ {bound:g})"


# ----- Streamlit App -----
# Keep BLAS/OpenMP within the host's CPU quota (no-op without one).
limit_native_threads()
st.title("Auto-Detected Multi-Output Surrogate Model Grid Search & Report Generator")

uploaded_file = st.file_uploader(
    "Upload CSV file. Example files can be found in the repo: `multioutreg/docs/_static/example_datasets/`.",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview of Data:", df.head())

    # Missing value detection and per-column imputation choices
    _missing_summary = detect_missing(df)
    _impute_choices: Dict[str, str] = {}
    if not _missing_summary.empty:
        st.warning(
            f"{int(_missing_summary['missing_count'].sum())} missing value(s) detected "
            f"across {len(_missing_summary)} column(s)."
        )
        st.dataframe(_missing_summary)
        st.write("**Choose how to handle missing values per column:**")
        st.caption(
            "Missing target values are never imputed: those rows are always dropped. "
            "Imputation of inputs is fitted on the training split only."
        )
        for _col in _missing_summary.index:
            _impute_choices[_col] = st.selectbox(
                f"`{_col}`",
                ["Impute (KNN)", "Drop rows"],
                key=f"imp_{_col}",
            )

    # ── Time Series Suitability Check ──────────────────────────────
    with st.expander("Auto-detect time series structure (optional)", expanded=False):
        st.caption(
            "Run statistical tests (ADF, Ljung-Box) to determine whether your data "
            "has temporal autocorrelation and would benefit from time series models."
        )
        _numeric_01 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        _c1, _c2 = st.columns(2)
        with _c1:
            _ts_dt_01 = st.selectbox(
                "Datetime column (optional)",
                ["<none>"] + list(df.columns),
                key="ts_dt_01",
            )
        with _c2:
            _ts_tgt_01 = st.selectbox(
                "Target column to test",
                ["<select>"] + _numeric_01,
                key="ts_tgt_01",
            )
        if st.button("Check time series suitability", key="ts_check_01"):
            if _ts_tgt_01 == "<select>":
                st.info("Select a target column to run the check.")
            else:
                _dt_arg_01 = None if _ts_dt_01 == "<none>" else _ts_dt_01
                with st.spinner("Running statistical tests..."):
                    _ts_result_01 = check_ts_suitability(df, _ts_tgt_01, datetime_col=_dt_arg_01)
                _rc1, _rc2, _rc3 = st.columns(3)
                _rc1.metric("ADF p-value", f"{_ts_result_01.get('adf_pvalue', 'N/A')}")
                _rc2.metric("Ljung-Box p-value", f"{_ts_result_01.get('ljungbox_pvalue', 'N/A')}")
                _rc3.metric("Detected period", _ts_result_01.get("seasonal_period") or "None")
                if _ts_result_01.get("suitable"):
                    st.success(_ts_result_01["recommendation"])
                    try:
                        st.page_link(
                            "pages/05_Time_Series_Forecasting.py",
                            label="Go to Time Series Forecasting page",
                        )
                    except Exception:
                        st.info("Navigate to page 05 — Time Series Forecasting to use the pipeline.")
                else:
                    st.warning(_ts_result_01["recommendation"])
                with st.expander("Full suitability report"):
                    st.json(_ts_result_01)
    # ───────────────────────────────────────────────────────────────

    with st.expander("⚙️ Advanced Settings", expanded=False):
        # PCA
        use_pca = st.checkbox("Apply PCA to input features", key="use_pca")
        n_components = None
        pca_method = None
        pca_threshold = None
        if use_pca:
            max_comp = len(df.columns)
            pca_method = st.selectbox(
                "PCA component selection method",
                ["Manual", "Explained variance threshold", "Kaiser rule"],
                key="pca_method",
            )
            if pca_method == "Manual":
                n_components = st.number_input(
                    "Number of PCA components",
                    min_value=1,
                    max_value=max_comp,
                    value=min(2, max_comp),
                    step=1,
                    key="pca_n_components",
                )
            elif pca_method == "Explained variance threshold":
                pca_threshold = st.slider(
                    "Explained variance threshold",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.9,
                    step=0.01,
                    key="pca_threshold",
                )

        # Conformal prediction
        use_conformal = st.checkbox("Compute conformal prediction intervals", key="use_conformal")
        conformal_alpha_sel = st.slider(
            "Conformal alpha (miscoverage level)",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            disabled=not use_conformal,
            key="conformal_alpha",
        )

        # Bayesian Neural Network
        use_bnn = st.checkbox(
            "Include BNN (Bayesian Neural Network) as a joint multi-output candidate",
            value=False,
            disabled=not _torch_available,
            help="Requires PyTorch. Trains a Bayesian Neural Network with MC Dropout uncertainty and competes against per-output models.",
            key="use_bnn",
        )

        # Mixture of Experts
        use_moe = st.checkbox(
            "Include MoE (Mixture of Experts) as a joint multi-output candidate",
            value=False,
            help="Trains K expert regressors specialised to different input regions via a learned gating network.",
            key="use_moe",
        )
        moe_n_experts = 4
        moe_gating_type = "linear"
        if use_moe:
            moe_n_experts = st.slider(
                "MoE: Number of experts",
                min_value=2,
                max_value=8,
                value=4,
                step=1,
                key="moe_n_experts",
            )
            moe_gating_type = st.selectbox(
                "MoE: Gating type",
                ["linear", "mlp"],
                key="moe_gating_type",
            )

        skip_expensive = st.checkbox(
            "Skip computationally expensive models",
            # On by default where CPU is capped (e.g. the hosted demo).
            value=max_threads() is not None,
            help=(
                "Excludes Gaussian Process (O(N³)), Neural Network/MLP, and NGBoost "
                "from the grid search. Recommended for large datasets or quick exploratory runs."
            ),
            key="skip_expensive",
        )

        # Multi-objective regret settings & bounds
        st.markdown("**Multi-objective regret** (all targets treated as objectives to minimize)")
        _rcol_a, _rcol_b = st.columns(2)
        with _rcol_a:
            hv_ref_mode = st.selectbox(
                "Hypervolume reference point",
                ["Auto (10% beyond worst)", "Auto with margin (%)", "Custom"],
                key="hv_ref_mode",
            )
            if hv_ref_mode == "Auto with margin (%)":
                st.number_input(
                    "Reference margin (%)", min_value=0.0, max_value=100.0,
                    value=10.0, step=0.5, key="hv_margin_pct",
                )
            elif hv_ref_mode == "Custom":
                st.text_input(
                    "Custom reference point (one value per target, comma/space separated)",
                    placeholder="e.g., 3.0, 3.0",
                    key="hv_ref_custom",
                )
        with _rcol_b:
            weights_mode = st.selectbox(
                "Scalarized weights",
                ["Auto (Dirichlet k=8)", "Uniform", "Custom"],
                key="weights_mode",
            )
            if weights_mode == "Custom":
                st.text_area(
                    "Custom weight vectors (one per line, comma/space separated)",
                    placeholder="e.g.\n0.5, 0.5\n0.2, 0.8",
                    height=100,
                    key="weights_custom",
                )
        st.caption("Optional pass/fail bounds (leave blank to skip)")
        _bcol1, _bcol2, _bcol3 = st.columns(3)
        _bcol1.text_input("Max HV regret (≤)", key="bound_hv")
        _bcol2.text_input("Max scalarized regret (≤)", key="bound_scalar")
        _bcol3.text_input("Max ε-regret (≤)", key="bound_eps")

    with st.form("column_selection"):
        input_cols = st.multiselect("Select input features", options=df.columns)
        output_cols = st.multiselect("Select output targets", options=df.columns)

        description = st.text_area("Optional: Project description")
        report_styles = report_format_control()
        submitted = st.form_submit_button("Run Grid Search")

    if submitted and input_cols and output_cols:
        use_pca = st.session_state.use_pca
        pca_method = st.session_state.get("pca_method")
        n_components = st.session_state.get("pca_n_components")
        pca_threshold = st.session_state.get("pca_threshold")
        use_conformal = st.session_state.use_conformal
        conformal_alpha_sel = st.session_state.conformal_alpha
        use_bnn = st.session_state.use_bnn
        use_moe = st.session_state.use_moe
        moe_n_experts = st.session_state.get("moe_n_experts", 4)
        moe_gating_type = st.session_state.get("moe_gating_type", "linear")
        skip_expensive = st.session_state.skip_expensive

        # Apply imputation / row-dropping for the selected columns before training
        _all_selected_cols = list(input_cols) + list(output_cols)
        _cols_to_impute = [
            c for c in _all_selected_cols
            if st.session_state.get(f"imp_{c}") == "Impute (KNN)"
        ]
        _cols_to_drop = [
            c for c in _all_selected_cols
            if st.session_state.get(f"imp_{c}") == "Drop rows"
        ]
        _rows_before = len(df)
        # Targets are never imputed: rows missing a target are dropped. Inputs marked
        # "Impute (KNN)" are imputed after the train/test split (below).
        df = drop_unusable_rows(df, list(output_cols), _cols_to_drop)
        _rows_after = len(df)
        _imputation_summary = None
        if _cols_to_impute or _cols_to_drop:
            _summary_cols = []
            for _c in _cols_to_impute:
                if _c in _missing_summary.index:
                    _summary_cols.append({
                        "name": _c,
                        "action": (
                            "Rows dropped (missing target)" if _c in output_cols
                            else "Imputed (KNN, fit on training split)"
                        ),
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

        X = df[input_cols].values
        y = df[output_cols].values
        feature_names = list(input_cols)
        used_feature_names = feature_names

        # How performance is estimated and intervals calibrated depends on n:
        # nested CV when a 25% test split would be tiny, CV+ when a 20%
        # calibration split is too small for alpha (AUDIT S1, S2).
        plan = plan_evaluation(len(X), alpha=conformal_alpha_sel if use_conformal else None)
        for _note in plan.notes:
            st.info(_note)

        def _preprocess(X_tr, X_te):
            # KNN imputation and PCA are fitted on the training rows only.
            X_tr, X_te = impute_train_test(X_tr, X_te)
            pca = None
            if use_pca:
                pca = fit_input_pca(X_tr, X_te, pca_method, n_components, pca_threshold)
                X_tr, X_te = pca.X_train, pca.X_test
            return X_tr, X_te, pca

        _EXPENSIVE_DISPLAY_NAMES = {
            "gp": "Gaussian Process",
            "mlp": "Neural Network (MLP)",
            "ngboost": "NGBoost",
        }

        def _make_model():
            model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
            mo_candidates = []
            if use_bnn and _torch_available:
                from multioutreg.surrogates.bnn_pytorch import BNNSurrogate
                mo_candidates.append(
                    BNNSurrogate(hidden_layer_sizes=(128, 64), max_epochs=300, random_state=0)
                )
            if use_moe:
                from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
                mo_candidates.append(
                    MixtureOfExpertsSurrogate(
                        n_experts=moe_n_experts,
                        gating_type=moe_gating_type,
                        random_state=0,
                    )
                )
            if mo_candidates:
                model.register_multi_output_candidates(mo_candidates)
            if skip_expensive:
                drop_candidates(model, EXPENSIVE_MODEL_NAMES)
            return model

        if skip_expensive:
            _skipped = [n for n in AutoDetectMultiOutputRegressor.with_vendored_surrogates()._model_names
                        if n in EXPENSIVE_MODEL_NAMES]
            if _skipped:
                st.info(
                    f"Skipping {len(_skipped)} expensive model(s): "
                    + ", ".join(_EXPENSIVE_DISPLAY_NAMES.get(n, n) for n in _skipped)
                )

        with st.status("Selecting and evaluating models…", expanded=False) as _eval_status:
            evaluation = run_evaluation(
                _make_model, X, y, plan, _preprocess,
                on_step=lambda label: _eval_status.write(label + "…"))
            _eval_status.update(label="Model selection done", state="complete")
        y_test, best_pred, best_std = evaluation.y_true, evaluation.pred, evaluation.std
        X_train = evaluation.X_fit
        X_test = evaluation.X_eval
        best_model = model = evaluation.model
        best_combo = [m.__class__.__name__ for m in model.models_]

        # PCA.
        pca_variance_plot = None
        pca_explained_variance = None
        pca_n_components = None
        feature_names_pca = None
        kaiser_rule_suggestion = None
        if use_pca:
            _pca = evaluation.info
            pca_n_components = _pca.n_components
            feature_names_pca = _pca.feature_names
            pca_variance_plot = generate_pca_variance_plot(
                _pca.preview_pca,
                n_selected=pca_n_components,
                threshold=pca_threshold if pca_method == "Explained variance threshold" else None,
            )
            pca_explained_variance = _pca.preview_pca.explained_variance_ratio_.tolist()
            kaiser_rule_suggestion = (
                f"Kaiser rule suggests **{_pca.kaiser_k}** components "
                "(eigenvalues > 1 on standardized inputs)."
            )
            st.markdown(kaiser_rule_suggestion)
            st.image(
                f"data:image/png;base64,{pca_variance_plot}",
                caption="Scree Plot",
                width="stretch",
            )
            used_feature_names = feature_names_pca

        st.caption(f"Performance estimate: {plan.describe()}.")

        # Conformal prediction
        conformal_intervals = None
        if use_conformal:
            y_lower, y_upper = evaluation.lower, evaluation.upper
            conformal_intervals = (y_lower, y_upper)

            from multioutreg.conformal.metrics import conformal_summary as _conf_summary
            conf_df = _conf_summary(
                y_test, y_lower, y_upper,
                alpha=conformal_alpha_sel,
                output_names=list(output_cols),
            )
            st.write(f"### Conformal Prediction Summary (alpha={conformal_alpha_sel})")
            st.dataframe(conf_df)

        # # Show seaborn PairGrid plot with KDE in lower triangle
        # if df.shape[1] >= 2:
        #     st.write("### PairGrid Visualization (KDE Lower Triangle)")

        #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        #     if len(numeric_cols) >= 2:
        #         g = make_doe_plot(df=df, numeric_cols=numeric_cols)
        #         st.pyplot(g.fig)
        #         del g
        #         grid_plot = plot_surrogate_model_summary(
        #                 X_train=X_train,
        #                 X_test=X_test,
        #                 Y_train=y_train,
        #                 Y_test=y_test,
        #                 model=best_model,
        #                 savefig=False,
        #         )
        #         st.pyplot(grid_plot)
        #         del grid_plot
        #     else:
        #         st.info("Not enough numeric columns for pairwise plot.")

        st.write("### Selected Surrogates")
        st.json(best_combo)

        st.write("### Metrics Table")
        metrics = {}
        for i, name in enumerate(output_cols):
            y_true = y_test[:, i]
            y_pred = best_pred[:, i]
            metrics[name] = {
                "r2": r2_score(y_true, y_pred),
                "rmse": root_mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "mean_predicted_std": float(np.mean(best_std[:, i])),
            }
        st.dataframe(pd.DataFrame(metrics).T)

        uncertainty_metrics = gaussian_uncertainty_metrics(y_test, best_pred, best_std)
        st.write("### Uncertainty Calibration (Gaussian std, test set)")
        st.dataframe(pd.Series(uncertainty_metrics, name="value").to_frame())

        # Multi-objective decision regret: true test values at the candidates the predictions pick
        _m = y_test.shape[1]
        hv_ref_mode = st.session_state.hv_ref_mode
        weights_mode = st.session_state.weights_mode
        hv_margin_pct = None
        if hv_ref_mode == "Custom":
            try:
                _ref_vals = _parse_csv_floats(st.session_state.get("hv_ref_custom", ""))
            except ValueError:
                _ref_vals = []
            if len(_ref_vals) == _m:
                reference_point = np.asarray(_ref_vals, dtype=float)
            else:
                st.warning("Invalid custom reference point; falling back to auto 10%.")
                hv_margin_pct = 10.0
                reference_point = _auto_reference_point(y_test, hv_margin_pct)
        else:
            hv_margin_pct = (
                float(st.session_state.get("hv_margin_pct", 10.0))
                if hv_ref_mode == "Auto with margin (%)" else 10.0
            )
            reference_point = _auto_reference_point(y_test, hv_margin_pct)

        if _m == 1:
            regret_weights = 1.0
        elif weights_mode == "Custom":
            regret_weights = _parse_weight_matrix(st.session_state.get("weights_custom", ""), _m)
            if regret_weights is None:
                st.warning("Invalid custom weights; falling back to uniform.")
                regret_weights = np.full((1, _m), 1.0 / _m)
        elif weights_mode == "Uniform":
            regret_weights = np.full((1, _m), 1.0 / _m)
        else:
            regret_weights = np.random.default_rng(7).dirichlet(np.ones(_m), size=8)

        regret_metrics = {
            "hv": float(hypervolume_regret(y_test, best_pred, reference_point=reference_point)),
            "scalar": float(scalarized_regret(y_test, best_pred, weights=regret_weights, reduce="mean")),
            "eps": float(epsilon_regret(y_test, best_pred)),
        }
        regret_bounds = {
            k: _parse_optional_float(st.session_state.get(f"bound_{k}"))
            for k in ("hv", "scalar", "eps")
        }
        regret_settings = {
            "hv_ref_mode": hv_ref_mode,
            "hv_margin_pct": hv_margin_pct,
            "reference_point": reference_point.tolist(),
            "weights_mode": weights_mode,
            "weights_shape": None if np.isscalar(regret_weights) else list(np.shape(regret_weights)),
        }

        # Shap plots
        shap_img = safe_plot_b64(
            plot_multioutput_shap_bar_subplots,
            best_model,
            X_train,
            feature_names=used_feature_names,
            output_names=output_cols,
        )
        st.write("### SHAP Summary Plot")
        st.image(
            f"data:image/png;base64,{shap_img}",
            caption="Mean(|SHAP value|) for each feature and output",
            width="stretch",
        )

        html = render_styles(
            generate_html_report,
            report_styles,
            model_type="AutoDetectMultiOutputRegressor",
            fidelity_levels=[],
            output_names=output_cols,
            description=description,
            metrics=metrics,
            uncertainty_metrics=uncertainty_metrics,
            y_test=y_test,
            best_pred=best_pred,
            best_std=best_std,
            best_model=best_model,
            X_train=X_train,
            n_train=evaluation.n_train,
            n_test=evaluation.n_test,
            cross_validation=(f"Model selection: {best_model.cv_description_}. "
                              f"Performance estimate: {plan.describe()}"),
            seed=0,
            notes="Generated report.",
            feature_names=used_feature_names,
            pca_explained_variance=pca_explained_variance,
            pca_variance_plot=pca_variance_plot,
            pca_method=pca_method,
            pca_threshold=pca_threshold,
            pca_n_components=pca_n_components,
            kaiser_rule_suggestion=kaiser_rule_suggestion,
            shap_plot=shap_img,
            conformal_intervals=conformal_intervals if use_conformal else None,
            conformal_alpha=conformal_alpha_sel if use_conformal else None,
            calibration_residuals=evaluation.fit.residuals if use_conformal else None,
            imputation_summary=_imputation_summary,
            regret_metrics=regret_metrics,
            regret_bounds=regret_bounds,
            regret_settings=regret_settings,
            selection_scores=getattr(best_model, "selection_scores_", None),
        )

        # Stash results so the regret summary and downloads survive the rerun a download click triggers
        st.session_state["mor_artifacts"] = {
            "html_report": html,
            "model_bytes": _serialize_model(best_model),
            "regret_metrics": regret_metrics,
            "regret_bounds": regret_bounds,
            "regret_settings": regret_settings,
        }

    _arts = st.session_state.get("mor_artifacts")
    if _arts:
        st.write("### Multi-Objective Regret")
        st.caption(
            "Decision regret: the test rows on the model's predicted Pareto front are the "
            "picks, and each metric measures how much worse their true values are than the "
            "true Pareto front of the test set. Zero is ideal; over-promising predictions "
            "do not lower it."
        )
        for _key, _label in [
            ("hv", "Hypervolume regret"),
            ("scalar", "Scalarized regret (mean)"),
            ("eps", "ε-regret"),
        ]:
            st.write(_regret_status_line(
                _label, _arts["regret_metrics"].get(_key), _arts["regret_bounds"].get(_key)
            ))
        with st.expander("Regret settings summary", expanded=False):
            st.json(_arts["regret_settings"])

        report_downloads_all(_arts["html_report"], "model_report_auto.html", "download_report")
        st.download_button(
            "Download Trained Model (.joblib)",
            _arts["model_bytes"],
            file_name="multioutreg_best_model.joblib",
            mime="application/octet-stream",
            key="download_model",
        )
