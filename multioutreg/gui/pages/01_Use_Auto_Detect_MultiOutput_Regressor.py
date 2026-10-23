# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
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
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test
from multioutreg.utils.metric_utils import gaussian_uncertainty_metrics
from multioutreg.time_series.ts_suitability import check_ts_suitability

# # NOTE: NOT used...yet.
# from multioutreg.figures.doe_plots import make_doe_plot
# from multioutreg.figures.model_comparison import plot_surrogate_model_summary



from multioutreg.report.auto_detect_report import generate_html_report


# ----- Streamlit App -----
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
            help=(
                "Excludes Gaussian Process (O(N³)), Neural Network/MLP, and NGBoost "
                "from the grid search. Recommended for large datasets or quick exploratory runs."
            ),
            key="skip_expensive",
        )

    with st.form("column_selection"):
        input_cols = st.multiselect("Select input features", options=df.columns)
        output_cols = st.multiselect("Select output targets", options=df.columns)

        description = st.text_area("Optional: Project description")
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        X_train, X_test = impute_train_test(X_train, X_test)
        feature_names = list(input_cols)
        used_feature_names = feature_names

        # # Show seaborn PairGrid plot with KDE in lower triangle
        # if df.shape[1] >= 2:
        #     st.write("### PairGrid Visualization (KDE Lower Triangle)")

        #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        #     if len(numeric_cols) >= 2:
        #         g = make_doe_plot(df=df, numeric_cols=numeric_cols)
        #         st.pyplot(g.fig)
        #     else:
        #         st.info("Not enough numeric columns for pairwise plot.")

        # PCA.
        pca_variance_plot = None
        pca_explained_variance = None
        pca_n_components = None
        feature_names_pca = None
        kaiser_rule_suggestion = None
        if use_pca:
            _pca = fit_input_pca(X_train, X_test, pca_method, n_components, pca_threshold)
            X_train, X_test = _pca.X_train, _pca.X_test
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

        model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
        _mo_candidates = []
        if use_bnn and _torch_available:
            from multioutreg.surrogates.bnn_pytorch import BNNSurrogate
            _mo_candidates.append(
                BNNSurrogate(hidden_layer_sizes=(128, 64), max_epochs=300, random_state=0)
            )
        if use_moe:
            from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
            _mo_candidates.append(
                MixtureOfExpertsSurrogate(
                    n_experts=moe_n_experts,
                    gating_type=moe_gating_type,
                    random_state=0,
                )
            )
        if _mo_candidates:
            model.register_multi_output_candidates(_mo_candidates)

        _EXPENSIVE_MODEL_NAMES = {"gp", "mlp", "ngboost"}
        _EXPENSIVE_DISPLAY_NAMES = {
            "gp": "Gaussian Process",
            "mlp": "Neural Network (MLP)",
            "ngboost": "NGBoost",
        }
        if skip_expensive:
            _keep = [i for i, n in enumerate(model._model_names) if n not in _EXPENSIVE_MODEL_NAMES]
            _skipped = [model._model_names[i] for i in range(len(model._model_names)) if model._model_names[i] in _EXPENSIVE_MODEL_NAMES]
            model.estimators = [model.estimators[i] for i in _keep]
            model.param_spaces = [model.param_spaces[i] for i in _keep]
            model._surrogate_constructors = [model._surrogate_constructors[i] for i in _keep]
            model._model_names = [model._model_names[i] for i in _keep]
            if _skipped:
                st.info(
                    f"Skipping {len(_skipped)} expensive model(s): "
                    + ", ".join(_EXPENSIVE_DISPLAY_NAMES.get(n, n) for n in _skipped)
                )

        # Conformal calibration must use data disjoint from both training and the
        # test set used to report coverage, so hold out part of the training split.
        if use_conformal:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train, test_size=0.2, random_state=0
            )

        model.fit(X_train, y_train)
        best_pred, best_std = model.predict(X_test, return_std=True)
        best_model = model
        best_combo = [m.__class__.__name__ for m in model.models_]

        # Conformal prediction
        conformal_intervals = None
        if use_conformal:
            model.calibrate_conformal(X_cal, y_cal)
            y_lower, y_upper = model.predict_interval(X_test, alpha=conformal_alpha_sel)
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

        html = generate_html_report(
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
            n_train=X_train.shape[0],
            n_test=X_test.shape[0],
            cross_validation=f"Model selection: {best_model.cv_description_}",
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
            calibration_residuals=model._conformal_residuals if use_conformal else None,
            imputation_summary=_imputation_summary,
        )

        st.download_button("Download HTML Report", html, file_name="model_report_auto.html", mime="text/html")
