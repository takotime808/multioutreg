# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Pure logic functions (extracted for testability — no Streamlit calls)
# ------------------------------------------------------------------

def _run_ts_pipeline(
    df: pd.DataFrame,
    target_col: str,
    datetime_col: "str | None",
    freq: str,
    verbose: bool,
) -> dict:
    """Run ARIMA / SARIMA / LSTM pipeline on df.

    Returns
    -------
    dict with keys:
        "perf_df"    : pd.DataFrame of per-model performance metrics
        "best_model" : str name of the best model
        "results"    : list of (metadata_dict, data_series, model_obj)
        "error"      : str (only present if a fatal error occurred)
    """
    from multioutreg.time_series.ts_dynamic_fit import (
        ARIMA, SARIMA, LSTM, DataProcessor, Ranker,
    )

    processor = DataProcessor(verbose=verbose)
    _, future_pred = processor.load_data(df)
    processor.validate_data(
        datetime_col=datetime_col or df.columns[0],
        target_col=target_col,
    )
    processed_data = processor.preprocess_data(
        datetime_col=datetime_col or df.columns[0],
        target_col=target_col,
        freq=freq,
    )

    results = []
    for ModelClass, label, extra in [
        (ARIMA,  "ARIMA",  {"verbose": verbose}),
        (SARIMA, "SARIMA", {"verbose": verbose}),
        (LSTM,   "LSTM",   {"fut_pred": future_pred, "train_window": 10, "verbose": verbose}),
    ]:
        try:
            if label == "LSTM":
                m = ModelClass(processed_data, feature_column=target_col, **extra)
            else:
                m = ModelClass(processed_data, target_col, **extra)
            perf, data_obj, model_obj = m.run()
            results.append((perf, data_obj, model_obj))
        except Exception as exc:
            pass  # individual model failures are surfaced as missing rows

    if not results:
        return {"error": "All models failed. Check the data and column selection.", "results": []}

    perf_rows = [r[0]["performance"] for r in results]
    perf_df = pd.DataFrame(perf_rows)

    best_model = Ranker(
        df=pd.DataFrame([r[0] for r in results]),
        verbose=False,
    ).get_best()

    return {
        "perf_df": perf_df,
        "best_model": best_model,
        "results": results,
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
        "Train and compare ARIMA, SARIMA, and LSTM models on your data. "
        "The pipeline auto-selects the best model by RMSE / MAE / MAPE ranking."
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

        run_btn = st.button("Run Pipeline", key="run_ts_pipeline", type="primary")

        if run_btn:
            datetime_col_arg = None if dt_col == "<use index>" else dt_col
            with st.spinner("Running ARIMA / SARIMA / LSTM pipeline…"):
                try:
                    out = _run_ts_pipeline(df2, tgt_col, datetime_col_arg, freq, verbose)
                except Exception as exc:
                    out = {"error": str(exc), "results": []}

            if "error" in out:
                st.error(out["error"])
            else:
                st.write("### Model comparison")
                st.dataframe(out["perf_df"])
                st.success(f"Best model: **{out['best_model']}**")

                # Visualize best model (ARIMA/SARIMA only — LSTM plots aren't concise)
                best_idx = [r[0]["performance"]["Model"] for r in out["results"]].index(
                    out["best_model"]
                )
                _, best_data, best_model_obj = out["results"][best_idx]
                if out["best_model"] in ("ARIMA", "SARIMA"):
                    try:
                        from multioutreg.time_series.ts_dynamic_fit.src.visualize import visualize_model
                        plt.close("all")
                        visualize_model(best_model_obj, best_data, target_col=tgt_col)
                        for fig_num in plt.get_fignums():
                            st.pyplot(plt.figure(fig_num))
                        plt.close("all")
                    except Exception as exc:
                        st.info(f"Visualization skipped: {exc}")

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
