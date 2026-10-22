# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Any
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from jinja2 import Template


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
from multioutreg.figures.shap_multioutput import generate_shap_plot
from multioutreg.figures.pca_plots import generate_pca_variance_plot
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
from multioutreg.surrogates import MultiFidelitySurrogate, LinearRegressionSurrogate
from multioutreg.surrogates.conformal_network_sklearn import ConformalPredictionNetworkSurrogate
from multioutreg.surrogates.rfgp_sklearn import _RFFEstimator
from multioutreg.surrogates.polynomial_bayesian_ridge_sklearn import _PBREstimator
from multioutreg.surrogates.nystroem_gp_sklearn import _NystroemEstimator
from multioutreg.surrogates.ard_gp_sklearn import _ARDGPEstimator
from multioutreg.model_selection.screening import ModelScreener

try:
    from ngboost import NGBRegressor as _NGBRegressor
    _NGBOOST_AVAILABLE = True
except ImportError:
    _NGBOOST_AVAILABLE = False

from multioutreg.surrogates.gpx_smt import _GPXEstimator, _GPX_AVAILABLE
from multioutreg.surrogates.kpls_smt import _KPLSEstimator, _KPLS_AVAILABLE
from multioutreg.surrogates.lightgbm_sklearn import _LIGHTGBM_AVAILABLE, _LGBMRegressor
from multioutreg.surrogates.xgboost_sklearn import _XGBOOST_AVAILABLE, _XGBRegressor
from multioutreg.surrogates.catboost_sklearn import _CATBOOST_AVAILABLE, _CatBoostRegressor
from multioutreg.surrogates.sparse_gp_gpytorch import _GPYTORCH_AVAILABLE, _SGPREstimator
from sklearn.linear_model import ElasticNet as _ElasticNet, Lasso as _Lasso, QuantileRegressor as _QuantileRegressor
from multioutreg.time_series.ts_suitability import check_ts_suitability

# NOTE: NOT used...yet.
from multioutreg.figures.doe_plots import make_doe_plot
from multioutreg.figures.model_comparison import plot_surrogate_model_summary
from multioutreg.utils.imputation import detect_missing, apply_imputation


from multioutreg.gui.grid_search_core import (
    RandomForestWithUncertainty,
    ExtraTreesWithUncertainty,
    GradientBoostingWithUncertainty,
    KNeighborsRegressorWithUncertainty,
    BootstrapLinearRegression,
    PerTargetRegressorWithStd,
    generate_html_report,
)


# ----- Streamlit App -----
st.title("Multi-Output Surrogate Model Grid Search & Report Generator")

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
        _numeric_gss = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        _gc1, _gc2 = st.columns(2)
        with _gc1:
            _ts_dt_gss = st.selectbox(
                "Datetime column (optional)",
                ["<none>"] + list(df.columns),
                key="ts_dt_gss",
            )
        with _gc2:
            _ts_tgt_gss = st.selectbox(
                "Target column to test",
                ["<select>"] + _numeric_gss,
                key="ts_tgt_gss",
            )
        if st.button("Check time series suitability", key="ts_check_gss"):
            if _ts_tgt_gss == "<select>":
                st.info("Select a target column to run the check.")
            else:
                _dt_arg_gss = None if _ts_dt_gss == "<none>" else _ts_dt_gss
                with st.spinner("Running statistical tests..."):
                    _ts_result_gss = check_ts_suitability(df, _ts_tgt_gss, datetime_col=_dt_arg_gss)
                _gc1r, _gc2r, _gc3r = st.columns(3)
                _gc1r.metric("ADF p-value", f"{_ts_result_gss.get('adf_pvalue', 'N/A')}")
                _gc2r.metric("Ljung-Box p-value", f"{_ts_result_gss.get('ljungbox_pvalue', 'N/A')}")
                _gc3r.metric("Detected period", _ts_result_gss.get("seasonal_period") or "None")
                if _ts_result_gss.get("suitable"):
                    st.success(_ts_result_gss["recommendation"])
                    try:
                        st.page_link(
                            "pages/05_Time_Series_Forecasting.py",
                            label="Go to Time Series Forecasting page",
                        )
                    except Exception:
                        st.info("Navigate to page 05 — Time Series Forecasting to use the pipeline.")
                else:
                    st.warning(_ts_result_gss["recommendation"])
                with st.expander("Full suitability report"):
                    st.json(_ts_result_gss)
    # ───────────────────────────────────────────────────────────────

    with st.expander("⚙️ Advanced Settings", expanded=False):
        # PCA
        use_pca = st.checkbox("Apply PCA to input features", key="use_pca")
        pca_method = None
        n_components = None
        pca_threshold = None
        if use_pca:
            pca_method = st.selectbox(
                "PCA component selection method",
                ["Manual", "Explained variance threshold", "Kaiser rule"],
                key="pca_method",
            )
            if pca_method == "Manual":
                n_components = st.number_input(
                    "Number of PCA components",
                    min_value=1,
                    max_value=len(df.columns),
                    value=min(2, len(df.columns)),
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
        conformal_alpha_sel = 0.1
        if use_conformal:
            conformal_alpha_sel = st.slider(
                "Conformal alpha (miscoverage level)",
                min_value=0.01,
                max_value=0.5,
                value=0.1,
                step=0.01,
                disabled=not use_conformal,
                key="conformal_alpha",
            )

        # Model screening
        use_screening = st.checkbox(
            "Pre-screen models (skip expensive/unsuitable models)",
            help=(
                "Runs Breusch-Pagan (heteroscedasticity), Ramsey RESET (linearity), "
                "Shapiro-Wilk (normality), and RF–LR R² gain (non-linearity) tests "
                "before training.  Computationally heavy models (GP, quantile GB, "
                "heteroscedastic ensembles) are only included when the data "
                "characteristics justify them."
            ),
        )

        skip_expensive = st.checkbox(
            "Skip computationally expensive models",
            help=(
                "Excludes Gaussian Process (O(N³)), NGBoost, and the Conformal "
                "Prediction Network from the grid search. Recommended for large "
                "datasets or quick exploratory runs."
            ),
        )

        # Mixture of Experts
        use_moe = st.checkbox(
            "Include MoE (Mixture of Experts) as a joint multi-output candidate",
            value=False,
            help="Trains K expert regressors specialised to different input regions via a learned gating network. Evaluated alongside per-target models.",
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
            )
            moe_gating_type = st.selectbox(
                "MoE: Gating type",
                ["linear", "mlp"],
            )

    with st.form("column_selection"):
        input_cols = st.multiselect("Select input features", options=df.columns)
        output_cols = st.multiselect("Select output targets", options=df.columns)

        description = st.text_area("Optional: Project description")
        submitted = st.form_submit_button("Run Grid Search")

    # if submitted:
    #     use_pca = st.session_state.use_pca
    #     pca_method = st.session_state.get("pca_method")
    #     n_components = st.session_state.get("pca_n_components")
    #     pca_threshold = st.session_state.get("pca_threshold")

    if submitted and input_cols and output_cols:
        use_pca = st.session_state.use_pca
        pca_method = st.session_state.get("pca_method")
        n_components = st.session_state.get("pca_n_components")
        pca_threshold = st.session_state.get("pca_threshold")

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
        if _cols_to_impute or _cols_to_drop:
            df = apply_imputation(df, _cols_to_impute, _cols_to_drop)
        _rows_after = len(df)
        _imputation_summary = None
        if _cols_to_impute or _cols_to_drop:
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

        X = df[input_cols].values
        y = df[output_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        feature_names = list(input_cols)

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
            preview_pca = PCA().fit(X_train)
            kaiser_k = int(np.sum(preview_pca.explained_variance_ > 1))
            if pca_method == "Manual":
                pca_n_components = int(n_components)
            elif pca_method == "Explained variance threshold":
                cum = np.cumsum(preview_pca.explained_variance_ratio_)
                pca_n_components = int(np.searchsorted(cum, pca_threshold) + 1)
            else:  # Kaiser rule
                pca_n_components = max(1, kaiser_k)
            pca = PCA(n_components=pca_n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            feature_names_pca = [f"PC{i+1}" for i in range(pca_n_components)]
            pca_variance_plot = generate_pca_variance_plot(
                preview_pca,
                n_selected=pca_n_components,
                threshold=pca_threshold if pca_method == "Explained variance threshold" else None,
            )
            pca_explained_variance = preview_pca.explained_variance_ratio_.tolist()
            kaiser_rule_suggestion = f"Kaiser rule suggests **{kaiser_k}** components (eigenvalues > 1)."
            st.markdown(kaiser_rule_suggestion)
            st.image(
                f"data:image/png;base64,{pca_variance_plot}",
                caption="Scree Plot",
                width="stretch",
            )

        surrogate_defs = [
            ("gpr", GaussianProcessRegressor, {"alpha": [1e-4], "kernel": [RBF(), Matern(nu=1.5)]}),
            ("rf", RandomForestWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
            ("et", ExtraTreesWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
            ("gb", GradientBoostingWithUncertainty, {"alpha": [0.95], "n_estimators": [50]}),
            ("hgb", HistGradientBoostingRegressor, {"max_iter": [100, 200], "learning_rate": [0.05, 0.1]}),
            ("knn", KNeighborsRegressorWithUncertainty, {"n_neighbors": [3]}),
            ("blr", BootstrapLinearRegression, {"n_bootstraps": [20]}),
            ("svr", SVR, {"C": [1.0, 10.0], "gamma": ["scale", "auto"]}),
            ("dt", DecisionTreeRegressor, {"max_depth": [1, None]}),
            ("cpn", ConformalPredictionNetworkSurrogate, {"hidden_layer_sizes": [(64,), (128,)], "max_iter": [500]}),
            (
                "mfs_lr",
                lambda: MultiFidelitySurrogate(LinearRegressionSurrogate, ["default"]),
                {},
            ),
            ("bayesian_ridge", BayesianRidge, {}),
            ("rfgp", _RFFEstimator, {"n_components": [100, 500], "length_scale": [0.1, 1.0, 10.0], "kernel": ["rbf", "matern52"]}),
            ("pbr", _PBREstimator, {"degree": [2, 3], "interaction_only": [False, True]}),
            ("sgp", _NystroemEstimator, {"n_components": [50, 200], "gamma": [None, 0.1, 1.0]}),
            ("ard_gp", _ARDGPEstimator, {"alpha": [1e-6, 1e-2]}),
            ("elastic_net", _ElasticNet, {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.5, 0.75]}),
            ("lasso", _Lasso, {"alpha": [0.01, 0.1, 1.0]}),
            ("quantile", _QuantileRegressor, {"alpha": [0.1, 1.0]}),
        ]
        if _NGBOOST_AVAILABLE:
            surrogate_defs.append(
                ("ngb", _NGBRegressor, {"n_estimators": [100, 200], "verbose": [False]})
            )
        if _GPX_AVAILABLE:
            surrogate_defs.append(
                ("gpx", _GPXEstimator, {"corr": ["squar_exp", "matern52"], "poly": ["constant", "linear"]})
            )
        if _KPLS_AVAILABLE:
            surrogate_defs.append(
                ("kpls", _KPLSEstimator, {"n_comp": [2, 4], "corr": ["squar_exp", "matern52"]})
            )
        if _LIGHTGBM_AVAILABLE:
            surrogate_defs.append(
                ("lgbm", _LGBMRegressor, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "num_leaves": [31, 63]})
            )
        if _XGBOOST_AVAILABLE:
            surrogate_defs.append(
                ("xgb", _XGBRegressor, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "max_depth": [4, 6]})
            )
        if _CATBOOST_AVAILABLE:
            surrogate_defs.append(
                ("catboost", _CatBoostRegressor, {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05], "depth": [4, 6], "verbose": [False]})
            )
        if _GPYTORCH_AVAILABLE:
            surrogate_defs.append(
                ("sgpr", _SGPREstimator, {"n_inducing": [30, 50], "learning_rate": [0.05, 0.1]})
            )

        if use_screening:
            _screener = ModelScreener().fit(X_train, y_train)
            _eligible = _screener.screen_per_output(list(output_cols))
            # Keep a surrogate_def if not screened or if it passes for at least one output
            surrogate_defs = [
                d for d in surrogate_defs
                if d[0] not in _eligible.index or _eligible.loc[d[0]].any()
            ]
            skipped = [
                d[0] for d in [
                    ("gpr", None, None), ("rf", None, None), ("et", None, None),
                    ("gb", None, None), ("knn", None, None), ("blr", None, None),
                    ("svr", None, None), ("dt", None, None), ("cpn", None, None),
                    ("mfs_lr", None, None), ("ngb", None, None),
                    ("bayesian_ridge", None, None), ("rfgp", None, None),
                    ("pbr", None, None), ("sgp", None, None), ("gpx", None, None),
                    ("ard_gp", None, None), ("kpls", None, None),
                    ("hgb", None, None), ("lgbm", None, None), ("xgb", None, None),
                    ("elastic_net", None, None), ("lasso", None, None),
                    ("quantile", None, None), ("catboost", None, None),
                    ("sgpr", None, None),
                ]
                if d[0] in _eligible.index and not _eligible.loc[d[0]].any()
            ]
            if skipped:
                st.info(
                    f"Pre-screening skipped {len(skipped)} model(s): "
                    + ", ".join(skipped)
                )

        _EXPENSIVE_MODELS = {"gpr", "ngb", "cpn", "ard_gp", "gpx", "kpls", "sgpr"}
        _EXPENSIVE_DISPLAY_NAMES = {
            "gpr": "Gaussian Process",
            "ngb": "NGBoost",
            "cpn": "Conformal Prediction Network",
            "ard_gp": "ARD Gaussian Process",
            "gpx": "GPX (Rust GP)",
            "kpls": "KPLS (Kriging+PLS)",
            "sgpr": "Sparse GP (GPyTorch)",
        }
        if skip_expensive:
            _skipped = [d[0] for d in surrogate_defs if d[0] in _EXPENSIVE_MODELS]
            surrogate_defs = [d for d in surrogate_defs if d[0] not in _EXPENSIVE_MODELS]
            if _skipped:
                st.info(
                    f"Skipping {len(_skipped)} expensive model(s): "
                    + ", ".join(_EXPENSIVE_DISPLAY_NAMES.get(k, k) for k in _skipped)
                )

        configs = [(name, Est, params) for name, Est, grid in surrogate_defs for params in ParameterGrid(grid)]

        best_score = np.inf
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

        # Evaluate MoE as a joint multi-output candidate and replace best model if it wins
        if use_moe:
            from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate
            try:
                moe = MixtureOfExpertsSurrogate(
                    n_experts=moe_n_experts,
                    gating_type=moe_gating_type,
                    random_state=0,
                )
                moe.fit(X_train, y_train)
                moe_pred, moe_std = moe.predict(X_test, return_std=True)
                moe_score = mean_squared_error(y_test, moe_pred)
                if moe_score < best_score:
                    best_score = moe_score
                    best_pred = moe_pred
                    best_std = moe_std
                    best_model = moe
                    best_combo = [{
                        "model": "MixtureOfExpertsSurrogate",
                        "n_experts": moe_n_experts,
                        "gating_type": moe_gating_type,
                    }]
                    st.info("MoE outperformed all per-target models and was selected.")
            except Exception as exc:
                st.warning(f"MoE evaluation failed: {exc}")

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

        st.write("### Best Model Configuration")
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

        # Conformal prediction
        conformal_intervals = None
        if use_conformal:
            from multioutreg.conformal.base import BaseConformalPredictor
            from multioutreg.conformal.metrics import conformal_summary as _conf_summary

            # Compute residuals directly on test data (no sklearn clone needed)
            residuals = np.abs(y_test - best_pred)
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
                output_names=list(output_cols),
            )
            st.write(f"### Conformal Prediction Summary (alpha={conformal_alpha_sel})")
            st.dataframe(conf_df)

        html = generate_html_report(
            model_type="PerTargetRegressorWithStd",
            fidelity_levels=[],
            output_names=output_cols,
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
            conformal_intervals=conformal_intervals if use_conformal else None,
            conformal_alpha=conformal_alpha_sel if use_conformal else None,
            imputation_summary=_imputation_summary,
        )

        st.download_button("Download HTML Report", html, file_name="model_report.html", mime="text/html")