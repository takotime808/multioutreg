# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from multioutreg.utils.thread_limits import limit_native_threads

# ------------------------------------------------------------------
# Pure logic functions (extracted for testability — no Streamlit calls)
# ------------------------------------------------------------------

def _run_ts_pipeline(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: "str | None",
    freq: str,
    verbose: bool,
    horizon: "int | None" = None,
    folds: "int | None" = None,
    lstm_epochs: int = 100,
    on_step=None,
) -> dict:
    """Rank ARIMA / SARIMA / LSTM / naive out-of-sample on df (same code as the CLI).

    Returns
    -------
    dict with keys:
        "perf_df"    : pd.DataFrame of per-model rolling-origin metrics, best first
        "best_model" : str name of the best model (lowest mean MASE)
        "ranking"    : RankingResult (plan, scores, Diebold-Mariano, forecast)
        "series"     : the resampled target series
        "validation" : dict returned by DataProcessor.validate_data
        "failures"   : dict mapping model label -> error message
        "error"      : str (only present if a fatal error occurred)
    """
    from multioutreg.time_series.ts_dynamic_fit import DataProcessor
    from multioutreg.time_series.ts_dynamic_fit.src.evaluate import rank_forecasters

    processor = DataProcessor(verbose=verbose)
    processor.load_data(df)
    validation = processor.validate_data(
        datetime_col=datetime_col or df.columns[0],
        target_col=target_col,
    )
    processed_data = processor.preprocess_data(
        datetime_col=datetime_col or df.columns[0],
        target_col=target_col,
        freq=freq,
    )
    series = processed_data[target_col]

    try:
        ranking = rank_forecasters(
            series.to_numpy(dtype=float), horizon=horizon, folds=folds,
            lstm_epochs=lstm_epochs, on_step=on_step)
    except (ValueError, RuntimeError) as exc:
        return {"error": str(exc), "results": [], "validation": validation, "failures": {}}

    perf_df = pd.DataFrame([
        {k: v for k, v in score.as_dict().items() if k not in ("per_origin_MASE", "error")}
        for score in ranking.scores
    ])
    return {
        "perf_df": perf_df,
        "best_model": ranking.best_model,
        "ranking": ranking,
        "series": series,
        "validation": validation,
        "failures": ranking.failures,
    }


def _run_surrogate_forecast(
    df: pd.DataFrame,
    target_col: str,
    n_lags: int,
    horizon: int,
    uncertainty: str,
    surrogate_name: str,
) -> dict:
    """Fit a LagFeatureForecaster with the chosen surrogate and run walk-forward CV.

    Parameters
    ----------
    df : pd.DataFrame — must contain ``target_col``
    target_col : str
    n_lags : int
    horizon : int
    uncertainty : str — "conformal", "return_std", or "none"
    surrogate_name : str — key in SURROGATE_OPTIONS

    Returns
    -------
    dict with keys:
        "forecast_result" : ForecastResult
        "cv_summary"      : dict from WalkForwardCV.summary()
        "error"           : str (only present on failure)
    """
    from multioutreg.time_series.lag_forecaster import LagFeatureForecaster
    from multioutreg.time_series.cv import WalkForwardCV

    surrogate = _build_surrogate(surrogate_name)
    series = df[target_col].dropna().to_numpy(dtype=float)

    lff = LagFeatureForecaster(
        surrogate=surrogate,
        n_lags=n_lags,
        horizon=horizon,
        uncertainty=uncertainty,
    )

    # Walk-forward CV (use shorter folds to stay fast in the GUI)
    min_train = max(n_lags + horizon + 5, int(0.6 * len(series)))
    cv = WalkForwardCV(min_train=min_train, horizon=horizon, step=max(1, horizon))
    cv_results = cv.evaluate(series, lff)
    cv_summary = cv.summary(cv_results)

    # Final fit on full series for the forecast
    lff.fit(series)
    forecast_result = lff.predict(horizon=horizon, quantiles=(0.1, 0.5, 0.9))

    return {
        "forecast_result": forecast_result,
        "cv_summary": cv_summary,
        "history": series,
    }


def _run_chronos_forecast(
    series_dict: dict,
    model_name: str,
    horizon: int,
    quantiles: "list[float]",
) -> "ForecastResult":
    """Run ChronosForecaster.fit().predict() — pure function for testability.

    Parameters
    ----------
    series_dict : dict mapping series name → np.ndarray
    model_name  : str (e.g., "amazon/chronos-bolt-tiny")
    horizon     : int
    quantiles   : list of float

    Returns
    -------
    ForecastResult
    """
    from multioutreg.time_series.chronos_adapter import ChronosForecaster
    forecaster = ChronosForecaster(model_name=model_name)
    forecaster.fit(series_dict)
    return forecaster.predict(prediction_length=horizon, quantiles=quantiles)


def _run_prophet_forecast(
    series: "np.ndarray",
    horizon: int,
    quantiles: "list[float]",
    seasonality_mode: str = "additive",
) -> "ForecastResult":
    """Fit ProphetForecaster and return a ForecastResult — pure function for testability.

    Parameters
    ----------
    series          : 1D np.ndarray of float values
    horizon         : int — prediction horizon in steps
    quantiles       : list of float quantile levels
    seasonality_mode : str — "additive" or "multiplicative"

    Returns
    -------
    ForecastResult
    """
    from multioutreg.time_series.prophet_adapter import ProphetForecaster
    forecaster = ProphetForecaster(seasonality_mode=seasonality_mode)
    forecaster.fit(series)
    return forecaster.predict(prediction_length=horizon, quantiles=quantiles)


def _run_neural_forecast(
    series: "np.ndarray",
    model_type: str,
    horizon: int,
    quantiles: "list[float]",
    input_size: int = 24,
    max_steps: int = 200,
) -> "ForecastResult":
    """Fit NeuralForecaster (N-BEATS / N-HiTS) — pure function for testability.

    Parameters
    ----------
    series      : 1D np.ndarray of float values
    model_type  : str — "nbeats" or "nhits"
    horizon     : int
    quantiles   : list of float
    input_size  : int
    max_steps   : int — reduced default for GUI responsiveness

    Returns
    -------
    ForecastResult
    """
    from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
    forecaster = NeuralForecaster(
        model_type=model_type,
        input_size=input_size,
        max_steps=max_steps,
    )
    forecaster.fit(series)
    return forecaster.predict(prediction_length=horizon, quantiles=quantiles)


# Surrogate options registry
def _build_surrogate(name: str):
    """Instantiate a surrogate by short name."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    options = {
        "linear": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=0),
        "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=0),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=0),
        "knn": KNeighborsRegressor(n_neighbors=5),
    }

    try:
        import lightgbm as lgb
        options["lightgbm"] = lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
    except ImportError:
        pass

    try:
        import xgboost as xgb
        options["xgboost"] = xgb.XGBRegressor(n_estimators=100, verbosity=0)
    except ImportError:
        pass

    if name not in options:
        raise ValueError(f"Unknown surrogate: {name!r}. Options: {list(options)}")
    return options[name]


SURROGATE_DISPLAY_NAMES = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "knn": "K-Nearest Neighbors",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

# Keep BLAS/OpenMP within the host's CPU quota (no-op without one).
limit_native_threads()
st.set_page_config(page_title="Time-Series Forecasting", layout="wide")
st.title("Time-Series Forecasting")

tab1, tab2, tab3 = st.tabs([
    "Chronos (Zero-Shot)",
    "ARIMA / SARIMA / LSTM",
    "Surrogate Forecasters",
])

# ──────────────────────────────────────────────────────────────────
# Tab 1: Chronos zero-shot forecasting
# ──────────────────────────────────────────────────────────────────
with tab1:
    st.caption(
        "Probabilistic forecasting using Chronos (zero-shot), "
        "Prophet (trend + seasonality), or NeuralForecast (N-BEATS / N-HiTS)."
    )

    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="chronos_upload")

    # Build model registry — only show adapters that are installed
    _CHRONOS_MODELS = [
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
        "amazon/chronos-t5-small",
    ]
    _model_options: "list[str]" = list(_CHRONOS_MODELS)
    try:
        import prophet as _prophet_pkg  # noqa: F401
        _model_options += ["prophet"]
    except ImportError:
        pass
    try:
        import neuralforecast as _nf_pkg  # noqa: F401
        _model_options += ["nbeats", "nhits"]
    except ImportError:
        pass

    _MODEL_DISPLAY = {
        "amazon/chronos-bolt-tiny": "Chronos-Bolt Tiny (zero-shot)",
        "amazon/chronos-bolt-small": "Chronos-Bolt Small (zero-shot)",
        "amazon/chronos-bolt-base": "Chronos-Bolt Base (zero-shot)",
        "amazon/chronos-t5-small": "Chronos-T5 Small (zero-shot)",
        "prophet": "Prophet (trend + seasonality)",
        "nbeats": "N-BEATS (NeuralForecast)",
        "nhits": "N-HiTS (NeuralForecast)",
    }

    model = st.selectbox(
        "Model",
        _model_options,
        format_func=lambda m: _MODEL_DISPLAY.get(m, m),
        key="chronos_model",
    )
    horizon_c = st.number_input("Prediction horizon (steps)", min_value=1, value=24, key="chronos_horizon")
    q_text = st.text_input("Quantiles (comma-separated)", "0.1,0.5,0.9", key="chronos_q")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        time_col = st.selectbox("Time column (optional)", ["<none>"] + list(df.columns), key="chronos_tc")
        numeric = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        targets = st.multiselect(
            "Target columns", numeric,
            default=(numeric[:1] if numeric else []),
            key="chronos_tgt",
        )
        go = st.button("Run forecast", key="chronos_run")

        if go and targets:
            try:
                from multioutreg.time_series.figures import plot_forecast_result

                q_list = [float(x) for x in q_text.split(",") if x.strip()]
                h = int(horizon_c)

                if model in _CHRONOS_MODELS:
                    series_dict = {c: df[c].dropna().to_numpy() for c in targets}
                    with st.spinner("Running Chronos…"):
                        res = _run_chronos_forecast(series_dict, model, h, q_list)
                    for i, sid in enumerate(res.ids):
                        fig = plot_forecast_result(
                            res, history=series_dict[sid], series_idx=i, title=sid
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                elif model == "prophet":
                    for col in targets:
                        series = df[col].dropna().to_numpy()
                        with st.spinner(f"Running Prophet on {col}…"):
                            res = _run_prophet_forecast(series, h, q_list)
                        fig = plot_forecast_result(res, history=series, title=col)
                        st.pyplot(fig)
                        plt.close(fig)

                elif model in ("nbeats", "nhits"):
                    for col in targets:
                        series = df[col].dropna().to_numpy()
                        with st.spinner(f"Running {_MODEL_DISPLAY[model]} on {col}…"):
                            res = _run_neural_forecast(series, model, h, q_list)
                        fig = plot_forecast_result(res, history=series, title=col)
                        st.pyplot(fig)
                        plt.close(fig)

            except Exception as exc:
                st.error(f"Forecast error: {exc}")

# ──────────────────────────────────────────────────────────────────
# Tab 2: ARIMA / SARIMA / LSTM pipeline
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.caption(
        "Compare ARIMA, SARIMA, LSTM and a naive baseline on your data. Every model is "
        "refitted at several rolling forecast origins and scored on the following, unseen "
        "points on the original scale; the best has the lowest mean MASE."
    )

    uploaded2 = st.file_uploader("Upload a CSV", type=["csv"], key="ts_pipeline_upload")

    if uploaded2:
        df2 = pd.read_csv(uploaded2)
        st.write("### Data preview")
        st.dataframe(df2.head())

        numeric_cols2 = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
        all_cols2 = list(df2.columns)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            dt_col = st.selectbox(
                "Datetime column",
                ["<use index>"] + all_cols2,
                key="ts_dt2",
            )
        with col_b:
            tgt_col = st.selectbox(
                "Target column",
                numeric_cols2 if numeric_cols2 else all_cols2,
                key="ts_tgt2",
            )
        with col_c:
            freq = st.selectbox(
                "Resampling frequency",
                ["1D", "1H", "15min", "1W"],
                key="ts_freq2",
            )
        with col_d:
            verbose = st.checkbox("Verbose logging", key="ts_verb2")

        col_e, col_f, col_g = st.columns(3)
        with col_e:
            horizon2 = st.number_input(
                "Forecast horizon (0 = auto)", min_value=0, value=0, step=1, key="ts_h2",
                help="Steps ahead to score. Auto: the seasonal period if one is detected, else 1.")
        with col_f:
            folds2 = st.number_input(
                "Rolling origins (0 = auto)", min_value=0, value=0, step=1, key="ts_folds2",
                help="Auto: every origin in the last 20% of the series, at most 10.")
        with col_g:
            epochs2 = st.number_input(
                "LSTM epochs", min_value=1, value=100, step=10, key="ts_epochs2",
                help="The LSTM is refitted at every origin, so this sets most of the run time.")

        run_btn = st.button("Run Pipeline", key="run_ts_pipeline", type="primary")

        if run_btn:
            datetime_col_arg = None if dt_col == "<use index>" else dt_col
            with st.status("Ranking models out-of-sample…", expanded=False) as _ts_status:
                try:
                    out = _run_ts_pipeline(
                        df2, tgt_col, datetime_col_arg, freq, verbose,
                        horizon=int(horizon2) or None, folds=int(folds2) or None,
                        lstm_epochs=int(epochs2),
                        on_step=lambda label: _ts_status.write(label + "…"))
                except Exception as exc:
                    out = {"error": str(exc), "results": []}
                _ts_status.update(label="Done", state="complete")

            if out.get("validation"):
                with st.expander("Validation report"):
                    st.json(out["validation"])
            for label, msg in out.get("failures", {}).items():
                st.warning(f"{label} failed: {msg}")

            if "error" in out:
                st.error(out["error"])
            else:
                ranking = out["ranking"]
                st.write("### Model comparison (out-of-sample)")
                st.caption(f"Evaluation: {ranking.plan.describe()}. Ranked by mean MASE "
                           "(below 1 beats a naive forecast's in-sample error).")
                st.dataframe(out["perf_df"])
                st.success(f"Best model: **{out['best_model']}**")
                dm = ranking.diebold_mariano
                if dm and dm.get("reason"):
                    st.caption(f"Diebold–Mariano ({dm['model_a']} vs {dm['model_b']}): "
                               f"not computed, {dm['reason']}.")
                elif dm and dm.get("n", 0) >= 2:
                    st.caption(
                        f"Diebold–Mariano ({dm['model_a']} vs {dm['model_b']}, HLN-corrected): "
                        f"statistic {dm['statistic']:.3f}, p = {dm['p_value']:.3f}. "
                        "Reported only; with few origins it has little power.")

                # History and the winner's next-H forecast (refitted on the full series).
                series = out["series"]
                if ranking.forecast is not None:
                    try:
                        future_index = pd.date_range(
                            series.index[-1], periods=len(ranking.forecast) + 1,
                            freq=pd.infer_freq(series.index) or freq)[1:]
                    except Exception:
                        future_index = range(len(series), len(series) + len(ranking.forecast))
                    chart = pd.concat([
                        series.rename("history").to_frame(),
                        pd.Series(ranking.forecast, index=future_index,
                                  name=f"{out['best_model']} forecast").to_frame(),
                    ], axis=1)
                    st.line_chart(chart)

# ──────────────────────────────────────────────────────────────────
# Tab 3: Surrogate Forecasters (LagFeatureForecaster + WalkForwardCV)
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.caption(
        "Use any of the 35+ surrogate models as a time-series forecaster via lag features. "
        "Walk-forward cross-validation is run automatically to report SMAPE and MASE."
    )

    uploaded3 = st.file_uploader("Upload a CSV", type=["csv"], key="surr_upload")

    if uploaded3:
        df3 = pd.read_csv(uploaded3)
        st.write("### Data preview")
        st.dataframe(df3.head())

        numeric_cols3 = [c for c in df3.columns if pd.api.types.is_numeric_dtype(df3[c])]

        col1, col2, col3 = st.columns(3)
        with col1:
            tgt_col3 = st.selectbox(
                "Target column",
                numeric_cols3 if numeric_cols3 else list(df3.columns),
                key="surr_tgt",
            )
        with col2:
            n_lags3 = st.slider("Lag window (n_lags)", min_value=4, max_value=48, value=12, key="surr_lags")
        with col3:
            horizon3 = st.number_input("Forecast horizon", min_value=1, max_value=30, value=5, key="surr_h")

        col4, col5 = st.columns(2)
        with col4:
            # Show only surrogates that are available
            available_keys = list(SURROGATE_DISPLAY_NAMES.keys())
            try:
                import lightgbm
            except ImportError:
                available_keys = [k for k in available_keys if k != "lightgbm"]
            try:
                import xgboost
            except ImportError:
                available_keys = [k for k in available_keys if k != "xgboost"]

            surrogate_key = st.selectbox(
                "Surrogate model",
                options=available_keys,
                format_func=lambda k: SURROGATE_DISPLAY_NAMES.get(k, k),
                key="surr_model",
            )
        with col5:
            uncertainty = st.radio(
                "Uncertainty method",
                options=["return_std", "none"],
                format_func=lambda x: {
                    "return_std": "Gaussian (predict std)",
                    "none": "Point prediction only",
                }[x],
                key="surr_unc",
            )

        run_surr = st.button("Run Surrogate Forecast", key="run_surr", type="primary")

        if run_surr:
            with st.spinner(f"Fitting {SURROGATE_DISPLAY_NAMES.get(surrogate_key, surrogate_key)} + walk-forward CV…"):
                try:
                    out3 = _run_surrogate_forecast(
                        df3, tgt_col3,
                        n_lags=int(n_lags3),
                        horizon=int(horizon3),
                        uncertainty=uncertainty,
                        surrogate_name=surrogate_key,
                    )
                    error3 = None
                except Exception as exc:
                    out3 = None
                    error3 = str(exc)

            if error3:
                st.error(f"Surrogate forecast error: {error3}")
            else:
                st.write("### Walk-Forward CV Summary")
                cv_s = out3["cv_summary"]
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Mean SMAPE", f"{cv_s['mean_smape']:.2f} ± {cv_s['std_smape']:.2f}")
                col_m2.metric("Mean MASE", f"{cv_s['mean_mase']:.3f} ± {cv_s['std_mase']:.3f}")
                col_m3.metric("Folds", str(cv_s["n_folds"]))

                from multioutreg.time_series.figures import plot_forecast_result
                fig = plot_forecast_result(
                    out3["forecast_result"],
                    history=out3["history"][-50:],   # show last 50 points
                    title=f"{SURROGATE_DISPLAY_NAMES.get(surrogate_key, surrogate_key)} — {tgt_col3}",
                )
                st.pyplot(fig)
                plt.close(fig)
