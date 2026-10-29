# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from typing import Dict, List, Tuple, Union, Any
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
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

from multioutreg.utils.thread_limits import limit_native_threads, max_threads
from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.shap_multioutput import generate_shap_plot
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.utils.pca_utils import fit_input_pca
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.utils.metric_utils import gaussian_uncertainty_metrics
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
from multioutreg.model_selection.evaluation import plan_evaluation, run_evaluation

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
from multioutreg.figures.doe_plots import make_doe_plot, pair_plot_b64
from multioutreg.figures.model_comparison import plot_surrogate_model_summary
from multioutreg.utils.imputation import detect_missing, drop_unusable_rows, impute_train_test


from multioutreg.gui.background import submit as submit_background, warm_up
from multioutreg.gui.report_controls import report_format_control
from multioutreg.report.render import render_styles
from multioutreg.report.build import build_report_data
from multioutreg.gui.input_checks import check_run_inputs
from multioutreg.gui.results_tabs import (
    compute_explanations,
    fill_report_tab,
    refresh_explain_tab,
    render_results,
    show_report_pending,
)
from multioutreg.gui.grid_search_core import (
    RandomForestWithUncertainty,
    ExtraTreesWithUncertainty,
    GradientBoostingWithUncertainty,
    KNeighborsRegressorWithUncertainty,
    BootstrapLinearRegression,
    PerTargetRegressorWithStd,
    build_model_bundle,
    build_model_pipeline,
    default_surrogate_defs,
    generate_html_report,
    EXPENSIVE_SURROGATES,
    PerTargetGridSearch,
    serialize_model_bundle,
)


SAMPLE_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "example_dataset_3_outputs.csv")
SAMPLE_DESCRIPTION = (
    "300 rows of synthetic data with 5 inputs (feature_0 to feature_4) "
    "and 3 outputs (pressure, temperature, velocity)."
)


@st.cache_data(show_spinner=False)
def _load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def _start_pair_plot(df: pd.DataFrame, source_id: Any, inputs: List[str], outputs: List[str]):
    """Start drawing the pair plot in a worker process, unless it is already running."""
    key = (source_id, tuple(inputs), tuple(outputs))
    current = st.session_state.get("gs_pair_job")
    if current is None or current[0] != key:
        cols = list(dict.fromkeys(list(inputs) + list(outputs)))
        job = submit_background("plots", pair_plot_b64, df[cols], list(inputs), list(outputs))
        st.session_state["gs_pair_job"] = (key, job)


class _PairPlotSlot:
    """The page's pair-plot placeholder, filled once the background job finishes.

    :meth:`refresh` is cheap to call often (e.g. on every progress tick): it
    draws the image once, and while the job runs it updates a waiting message
    once a second, which also gives Streamlit a chance to stop the run when the
    user clicks something.
    """

    def __init__(self):
        self._slot = st.empty()
        self._shown = False
        self._start = time.perf_counter()
        self._secs = -1

    def refresh(self) -> bool:
        """Show the plot if it's ready; return True when there is nothing left to wait for."""
        current = st.session_state.get("gs_pair_job")
        if current is None or self._shown:
            return True
        job = current[1]
        if not job.done():
            secs = int(time.perf_counter() - self._start)
            if secs != self._secs:
                self._slot.info(f"Drawing a pair plot of the data in the background ({secs}s)…")
                self._secs = secs
            return False
        result = job.result()
        with self._slot.container():
            with st.expander("Pair plot of the data", expanded=True):
                st.caption(
                    "Every input and output against every other: scatter above the "
                    "diagonal, 2-D histogram below it, each column's distribution on the "
                    "diagonal. Input labels are blue, output labels orange. The top-left "
                    "block shows how the inputs were sampled; the rows below it show which "
                    "inputs drive each output."
                )
                if result["img_b64"]:
                    st.image(f"data:image/png;base64,{result['img_b64']}", width="stretch")
                if result["note"]:
                    st.caption(result["note"])
        self._shown = True
        return True


# ----- Streamlit App -----
# Keep BLAS/OpenMP within the host's CPU quota (no-op without one).
limit_native_threads()
st.title("Multi-Output Surrogate Model Grid Search & Report Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# The sample dataset lets new users try the app without data of their own;
# an upload always takes precedence over it.
if uploaded_file is not None:
    st.session_state["gs_use_sample"] = False
elif not st.session_state.get("gs_use_sample", False):
    # A callback runs before the script, so the button is gone on the very run
    # that loads the sample.
    st.button(
        "Try with sample data",
        help="300 rows of synthetic data: 5 input features and 3 outputs "
             "(pressure, temperature, velocity).",
        on_click=st.session_state.update, kwargs={"gs_use_sample": True},
    )
_using_sample = uploaded_file is None and st.session_state.get("gs_use_sample", False)
data_source = SAMPLE_DATA_PATH if _using_sample else uploaded_file

if data_source:
    df = _load_csv(data_source)
    # Start the report workers now: their imports (UMAP, SHAP) then overlap
    # with choosing columns and fitting models.
    warm_up("report")

    # Results live in session_state so they survive reruns (e.g. clicking the
    # download button); drop them when a different file is uploaded.
    _source_id = getattr(data_source, "file_id", data_source)
    if st.session_state.get("gs_source_id") != _source_id:
        for _k in ("gs_results", "gs_pair_job"):
            st.session_state.pop(_k, None)
        st.session_state["gs_source_id"] = _source_id

    _sample_outputs = [c for c in df.columns if c.startswith("output_")] if _using_sample else []
    _sample_inputs = [c for c in df.columns if c not in _sample_outputs] if _using_sample else []
    if _using_sample:
        st.info(
            f"Using the sample dataset: {SAMPLE_DESCRIPTION} Each output is driven mainly by one input (pressure by "
            "feature_0, temperature by feature_2, velocity by feature_4), which the pair plot "
            "below makes visible. Inputs and outputs are pre-selected; click "
            "**Run Grid Search** to try it (about a minute with all models, or tick "
            "*Skip computationally expensive models* under Advanced Settings for a quicker "
            "run), or upload a CSV to use your own data."
        )
        _sc1, _sc2 = st.columns(2)
        with _sc1:
            with open(SAMPLE_DATA_PATH, "rb") as _fh:
                st.download_button(
                    "Download sample CSV", _fh.read(),
                    file_name=os.path.basename(SAMPLE_DATA_PATH), mime="text/csv",
                    help="See the expected format: one row per sample, one column per "
                         "input or output.",
                )
        with _sc2:
            st.button(
                "Stop using sample data",
                on_click=st.session_state.update, kwargs={"gs_use_sample": False},
            )
        # The columns are already known, so draw the pair plot right away.
        _start_pair_plot(df, _source_id, _sample_inputs, _sample_outputs)

    st.write("## Preview of Data:", df.head())
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns")

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
            # On by default where CPU is capped (e.g. the hosted demo).
            value=max_threads() is not None,
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
        input_cols = st.multiselect(
            "Select input features", options=df.columns, default=_sample_inputs)
        output_cols = st.multiselect(
            "Select output targets", options=df.columns, default=_sample_outputs)

        description = st.text_area(
            "Optional: Project description",
            value=f"Sample dataset: {SAMPLE_DESCRIPTION}" if _using_sample else "",
        )
        report_styles = report_format_control()
        submitted = st.form_submit_button("Run Grid Search")

        # Check the choices before any fitting, and show the problems in the form.
        _input_problems = []
        if submitted and input_cols and output_cols:
            _input_problems = check_run_inputs(
                df, list(input_cols), list(output_cols),
                conformal_alpha=conformal_alpha_sel if use_conformal else None,
                drop_row_cols=[c for c in list(input_cols) + list(output_cols)
                               if st.session_state.get(f"imp_{c}") == "Drop rows"],
            )
            for _p in _input_problems:
                if _p.kind in ("too_few_rows", "conformal_rows"):
                    st.warning(_p.message)
                else:
                    st.error(_p.message)

    # Below the form, so the Run button stays near the top. Filled by the
    # pair-plot job when it finishes, which may be in the middle of a run.
    _pair_slot = _PairPlotSlot()

    # if submitted:
    #     use_pca = st.session_state.use_pca
    #     pca_method = st.session_state.get("pca_method")
    #     n_components = st.session_state.get("pca_n_components")
    #     pca_threshold = st.session_state.get("pca_threshold")

    if submitted and not (input_cols and output_cols):
        st.warning("Select at least one input feature and one output target.")

    if submitted and input_cols and output_cols and not _input_problems:
        # Drawn in a worker process while the models fit; shown as soon as it's ready.
        _start_pair_plot(df, _source_id, list(input_cols), list(output_cols))
        _pair_slot.refresh()
        _notes: List[Tuple[str, str]] = []  # (st method, message), shown with the results
        _t_start = time.perf_counter()
        _status = st.status("Running grid search…", expanded=True)
        _status.write("Preparing data (missing values, train/test split)…")
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

        # How performance is estimated and intervals calibrated depends on n:
        # nested CV when a 25% test split would be tiny, CV+ when a 20%
        # calibration split is too small for alpha (AUDIT S1, S2).
        plan = plan_evaluation(len(X), alpha=conformal_alpha_sel if use_conformal else None)
        for _note in plan.notes:
            _notes.append(("info", _note))

        def _preprocess(X_tr, X_te):
            # KNN imputation and PCA are fitted on the training rows only.
            X_tr, X_te, imputer = impute_train_test(X_tr, X_te, return_imputer=True)
            pca = None
            if use_pca:
                pca = fit_input_pca(X_tr, X_te, pca_method, n_components, pca_threshold)
                X_tr, X_te = pca.X_train, pca.X_test
            return X_tr, X_te, {"imputer": imputer, "pca": pca}

        surrogate_defs = default_surrogate_defs()
        if skip_expensive:
            _skipped = [d[0] for d in surrogate_defs if d[0] in EXPENSIVE_SURROGATES]
            surrogate_defs = [d for d in surrogate_defs if d[0] not in EXPENSIVE_SURROGATES]
            if _skipped:
                _notes.append(("info",
                    f"Skipping {len(_skipped)} expensive model(s): "
                    + ", ".join(EXPENSIVE_SURROGATES.get(k, k) for k in _skipped)
                ))

        _moe_factory = None
        if use_moe:
            from multioutreg.surrogates.moe_surrogate import MixtureOfExpertsSurrogate

            def _moe_factory():
                return MixtureOfExpertsSurrogate(
                    n_experts=moe_n_experts, gating_type=moe_gating_type, random_state=0)

        # Configs are selected by CV on the training rows (pre-screening too,
        # when on); the evaluation rows are only used for the reported metrics.
        _prog = _status.progress(0.0, text="Fitting candidates…")
        _stage = {"label": ""}

        def _on_step(label: str) -> None:
            _stage["label"] = label
            _status.write(label + "…")

        def _on_progress(done: int, total: int, label: str) -> None:
            _prefix = f"{_stage['label']}: " if _stage["label"] else ""
            _prog.progress(done / total, text=f"{_prefix}{label} ({done + 1}/{total})")
            _pair_slot.refresh()

        try:
            evaluation = run_evaluation(
                lambda: PerTargetGridSearch(
                    surrogate_defs, moe_factory=_moe_factory,
                    on_progress=_on_progress, pre_screen=use_screening),
                X, y, plan, _preprocess, on_step=_on_step)
        except RuntimeError as exc:
            _status.update(label="Grid search failed", state="error")
            st.error(str(exc))
            st.stop()
        _prog.progress(1.0, text="Model selection done")
        _search = evaluation.model
        result = _search.result_
        best_model = result.model
        best_combo = result.combo
        y_test, best_pred, best_std = evaluation.y_true, evaluation.pred, evaluation.std
        X_train = evaluation.X_fit
        _imputer = evaluation.info["imputer"]
        if _search.skipped_:
            _notes.append(("info",
                f"Pre-screening skipped {len(_search.skipped_)} model(s): "
                + ", ".join(_search.skipped_)
            ))
        if result.moe_selected:
            _notes.append(("info", "MoE outperformed all per-target models and was selected."))
        if result.n_failed:
            _notes.append(("caption", f"{result.n_failed} candidate fit(s) failed and were skipped."))
        _status.write("Computing evaluation metrics…")

        # PCA.
        pca_variance_plot = None
        pca_explained_variance = None
        pca_n_components = None
        feature_names_pca = None
        kaiser_rule_suggestion = None
        _pca_transformer = None
        _pca_settings = None
        if use_pca:
            _pca = evaluation.info["pca"]
            pca_n_components = _pca.n_components
            _pca_transformer = _pca.transformer
            _pca_settings = {
                "standardize": True,
                "method": pca_method,
                "n_components": pca_n_components,
                "threshold": pca_threshold if pca_method == "Explained variance threshold" else None,
            }
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

        uncertainty_metrics = gaussian_uncertainty_metrics(y_test, best_pred, best_std)

        # Conformal prediction
        conformal_intervals = None
        conf_df = None
        q = None
        residuals = None
        if use_conformal:
            from multioutreg.conformal.base import BaseConformalPredictor
            from multioutreg.conformal.metrics import conformal_summary as _conf_summary

            # Intervals on the evaluation rows come from the plan (split
            # conformal or CV+). The exported model gets a fixed +/- q per
            # output: the conformal quantile of the calibration residuals
            # (out-of-fold residuals under CV+, which only approximates CV+).
            residuals = evaluation.fit.residuals
            q = np.array([
                BaseConformalPredictor._conformal_quantile(
                    residuals[:, j], conformal_alpha_sel
                )
                for j in range(residuals.shape[1])
            ])
            y_lower, y_upper = evaluation.lower, evaluation.upper
            conformal_intervals = (y_lower, y_upper)

            conf_df = _conf_summary(
                y_test, y_lower, y_upper,
                alpha=conformal_alpha_sel,
                output_names=list(output_cols),
            )

        # The tabs draw from the same numbers as the report (no SHAP/PDP here:
        # those come from the background job below).
        _report_data = build_report_data(
            output_names=list(output_cols), y_test=y_test, best_pred=best_pred,
            best_std=best_std, metrics=metrics, uncertainty_metrics=uncertainty_metrics,
            model_type=type(best_model).__name__, n_train=evaluation.n_train,
            n_test=evaluation.n_test, cross_validation=result.cv_description,
            conformal_intervals=conformal_intervals, conformal_alpha=(
                conformal_alpha_sel if use_conformal else None),
            calibration_residuals=residuals, grid_search_results=result.cv_table,
            explain=False,
        ).to_dict(include_images=False)
        _explain_job = submit_background(
            "plots", compute_explanations, best_model, X_train, list(output_cols),
            list(feature_names_pca if use_pca else feature_names))

        # The report's plots (prediction, SHAP, PDP, UMAP, ...) are drawn in a
        # worker process, so the results below show without waiting for them.
        _report_job = submit_background(
            "report",
            render_styles,
            generate_html_report,
            report_styles,
            model_type=type(best_model).__name__,
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
            cross_validation=(
                f"Model selected per output by {result.cv_description} on the training data. "
                f"Performance estimate: {plan.describe()}"
            ),
            seed=0,
            notes="Generated report.",
            feature_names=feature_names_pca if use_pca else feature_names,
            feature_names_pca=feature_names_pca,
            pca_explained_variance=pca_explained_variance,
            pca_variance_plot=pca_variance_plot,
            pca_method=pca_method,
            pca_threshold=pca_threshold,
            pca_n_components=pca_n_components,
            kaiser_rule_suggestion=kaiser_rule_suggestion,
            conformal_intervals=conformal_intervals if use_conformal else None,
            conformal_alpha=conformal_alpha_sel if use_conformal else None,
            calibration_residuals=residuals,
            imputation_summary=_imputation_summary,
            grid_search_results=result.cv_table,
        )

        # The exported pipeline replays every step the app applied to the inputs
        # (column order, KNN imputation, standardise + PCA) before the model.
        _bundle = build_model_bundle(
            build_model_pipeline(best_model, list(input_cols), _imputer, _pca_transformer),
            list(input_cols),
            list(output_cols),
            best_combo,
            pca_settings=_pca_settings,
            conformal_alpha=conformal_alpha_sel if use_conformal else None,
            conformal_quantiles=q,
            data_split={
                "evaluation": plan.scheme,
                "test_size": plan.test_size if plan.scheme == "holdout" else None,
                "outer_folds": plan.outer_cv.n_splits if plan.outer_cv is not None else None,
                "random_state": plan.random_state,
                "conformal_method": plan.conformal,
                "calibration_fraction": plan.calibration_size if plan.conformal == "split" else None,
                "model_selection_cv": result.cv_description,
            },
        )
        try:
            _model_bytes = serialize_model_bundle(_bundle)
        except Exception as exc:  # e.g. a surrogate holding an unpicklable handle
            _model_bytes = None
            _notes.append(("warning", f"The trained model could not be serialized: {exc}"))

        _status.update(
            label=(
                f"Grid search complete in {time.perf_counter() - _t_start:.1f}s; "
                "the HTML report is being built in the background (download it from "
                "the Report tab or under Export when it's ready)"
            ),
            state="complete",
            expanded=False,
        )

        st.session_state["gs_results"] = {
            "input_cols": list(input_cols),
            "output_cols": list(output_cols),
            "notes": _notes,
            "kaiser_rule_suggestion": kaiser_rule_suggestion,
            "pca_variance_plot": pca_variance_plot,
            "best_combo": best_combo,
            "metrics": pd.DataFrame(metrics).T,
            "uncertainty_metrics": pd.Series(uncertainty_metrics, name="value").to_frame(),
            "conformal_alpha": conformal_alpha_sel if use_conformal else None,
            "conformal_summary": conf_df,
            "html": None,
            "report_job": _report_job,
            "report_error": None,
            "model_bytes": _model_bytes,
            "model_config": json.dumps(_bundle["config"], indent=2),
            "report_data": _report_data,
            "cv_table": result.cv_table,
            "explain": None,
            "explain_job": _explain_job,
        }

    _res = st.session_state.get("gs_results")
    if _res is not None:
        if not submitted or _input_problems:
            st.caption(
                f"Showing the last run (inputs: {', '.join(_res['input_cols'])}; "
                f"outputs: {', '.join(_res['output_cols'])}). "
                "Click **Run Grid Search** to update."
            )
        for _kind, _msg in _res["notes"]:
            getattr(st, _kind)(_msg)
        _slots = render_results(_res)

        st.write("### Export the Report and Model")
        _slots["export_report"] = st.empty()
        if _res.get("model_bytes") is not None:
            st.download_button(
                "Download Trained Model (.joblib)",
                _res["model_bytes"],
                file_name="multioutreg_grid_search_model.joblib",
                mime="application/octet-stream",
                key="download_model",
            )
        st.download_button(
            "Download Model Config (.json)",
            _res["model_config"],
            file_name="multioutreg_grid_search_config.json",
            mime="application/json",
            key="download_model_config",
        )
        st.caption(
            "The .joblib file holds a fitted scikit-learn Pipeline with every step the "
            "app applied (input columns, KNN imputer, standardise + PCA if used, the "
            "selected model per output) plus the conformal quantiles. It contains the "
            "training inputs (the imputer and GP-type models keep them), needs the same "
            "multioutreg/scikit-learn versions to load (listed in the config) and, like "
            "any pickle, should only be loaded from a source you trust. The .json "
            "config describes the run so the model can be rebuilt and refitted."
        )
        with st.expander("Using the downloaded model", expanded=False):
            st.code(
                "import joblib\n"
                "import pandas as pd\n\n"
                'bundle = joblib.load("multioutreg_grid_search_model.joblib")\n'
                'pipe = bundle["pipeline"]\n'
                'df = pd.read_csv("new_data.csv")  # must contain bundle["input_columns"]\n'
                "mean, std = pipe.predict(df, return_std=True)\n"
                'pred = pd.DataFrame(mean, columns=bundle["output_columns"])\n\n'
                "# Conformal intervals, if they were computed\n"
                'if bundle["conformal"]:\n'
                '    q = pd.Series(bundle["conformal"]["quantiles"])\n'
                "    lower, upper = pred - q, pred + q\n",
                language="python",
            )

    # Wait for the background jobs at the very end, so everything above is
    # already on screen. The loop updates the page every second, which lets a
    # click interrupt it; the jobs keep running and the next run picks them up.
    _wait_start = time.perf_counter()
    _shown_secs = -1
    while True:
        _pair_ready = _pair_slot.refresh()
        _report_job = _res.get("report_job") if _res is not None else None
        _report_ready = _report_job is None or _report_job.done()
        _explain_job = _res.get("explain_job") if _res is not None else None
        if _explain_job is not None and _explain_job.done():
            try:
                _res["explain"] = _explain_job.result()
            except Exception as exc:
                _res["explain"] = {"shap": None, "pdp": None,
                                   "errors": [f"{type(exc).__name__}: {exc}"]}
            _res["explain_job"] = None
            refresh_explain_tab(_res, _slots)
            _explain_job = None
        if _pair_ready and _report_ready and _explain_job is None:
            break
        _secs = int(time.perf_counter() - _wait_start)
        if _report_job is not None and not _report_ready and _secs != _shown_secs:
            show_report_pending(_slots, _secs)
            _shown_secs = _secs
        time.sleep(0.25)

    if _res is not None:
        if _res.get("report_job") is not None:
            try:
                _res["html"] = _res["report_job"].result()
                st.toast("HTML report ready: download it under **Export** or in the "
                         "**Report** tab.", icon="📄")
            except Exception as exc:
                _res["report_error"] = f"Could not build the HTML report: {exc}"
            _res["report_job"] = None
        fill_report_tab(_res, _slots)
