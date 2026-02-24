# Copyright (c) 2025 takotime808

import os as _os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# sys.path injection so bare-import modules inside ts_dynamic_fit
# resolve correctly from anywhere (GUI, tests, CLI).
# ------------------------------------------------------------------
_TS_DIR = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "../../time_series/ts_dynamic_fit")
)
if _TS_DIR not in sys.path:
    sys.path.insert(0, _TS_DIR)

st.set_page_config(page_title="Time-Series Forecasting", layout="wide")
st.title("Time-Series Forecasting")

tab1, tab2 = st.tabs(["Chronos (Zero-Shot)", "ARIMA / SARIMA / LSTM"])

# ──────────────────────────────────────────────────────────────────
# Tab 1: Chronos zero-shot forecasting (original page, unchanged)
# ──────────────────────────────────────────────────────────────────
with tab1:
    from multioutreg.time_series.chronos_adapter import ChronosForecaster

    st.caption("Zero-shot probabilistic forecasting using Chronos-style foundation models.")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="chronos_upload")
    model = st.selectbox(
        "Model",
        [
            "amazon/chronos-bolt-tiny",
            "amazon/chronos-bolt-small",
            "amazon/chronos-bolt-base",
            "amazon/chronos-t5-small",
        ],
        key="chronos_model",
    )
    horizon = st.number_input("Prediction horizon (steps)", min_value=1, value=24, key="chronos_horizon")
    q_text = st.text_input("Quantiles (comma-separated)", "0.1,0.5,0.9", key="chronos_q")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        time_col = st.selectbox("Time column (optional)", ["<none>"] + list(df.columns), key="chronos_tc")
        numeric = [
            c for c in df.columns
            if c != time_col and pd.api.types.is_numeric_dtype(df[c])
        ]
        targets = st.multiselect("Target columns", numeric, default=(numeric[:1] if numeric else []), key="chronos_tgt")
        go = st.button("Run forecast", key="chronos_run")

        if go and targets:
            forecaster = ChronosForecaster(model_name=model)
            series = {c: df[c].dropna().to_numpy() for c in targets}
            res = forecaster.fit(series).predict(
                horizon, quantiles=[float(x) for x in q_text.split(",") if x.strip()]
            )
            for i, sid in enumerate(res.ids):
                hist = series[sid]
                fig = plt.figure()
                plt.plot(np.arange(len(hist)), hist, label="history")
                for qi, q in enumerate(res.q_levels):
                    plt.plot(
                        np.arange(len(hist), len(hist) + horizon),
                        res.quantiles[i, qi, :],
                        label=f"q{q}",
                    )
                plt.title(sid)
                plt.legend()
                st.pyplot(fig)
                plt.close(fig)

# ──────────────────────────────────────────────────────────────────
# Tab 2: ARIMA / SARIMA / LSTM pipeline (ts_dynamic_fit)
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.caption(
        "Train and compare ARIMA, SARIMA, and LSTM models on your data. "
        "The pipeline auto-selects the best model by RMSE / MAE / MAPE ranking."
    )

    # Check that ts_dynamic_fit modules are importable
    _ts_available = False
    _ts_import_error = None
    try:
        from data_handling.DataProcessor import DataProcessor
        from algs.arima import ARIMA
        from algs.sarima import SARIMA
        from algs.lstm import LSTM
        from src.ranker import Ranker
        from src.visualize import visualize_model
        _ts_available = True
    except Exception as _e:
        _ts_import_error = _e

    if not _ts_available:
        st.error(
            f"Could not import ts_dynamic_fit modules. "
            f"Make sure all dependencies (statsmodels, torch, joblib) are installed.\n\n"
            f"Error: {_ts_import_error}"
        )
    else:
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
                import tempfile

                datetime_col_arg = None if dt_col == "<use index>" else dt_col
                original_cwd = _os.getcwd()
                tmp_dir = tempfile.mkdtemp(prefix="ts_pipeline_")

                try:
                    # Pre-create directories that algs write to relative to CWD
                    _os.makedirs(_os.path.join(tmp_dir, "logs"), exist_ok=True)
                    _os.makedirs(_os.path.join(tmp_dir, "temp"), exist_ok=True)
                    _os.chdir(tmp_dir)

                    with st.spinner("Preprocessing data..."):
                        processor = DataProcessor(verbose=verbose)
                        _, future_pred = processor.load_data(df2)
                        validation_results = processor.validate_data(
                            datetime_col=datetime_col_arg or df2.columns[0],
                            target_col=tgt_col,
                        )
                        processed_data = processor.preprocess_data(
                            datetime_col=datetime_col_arg or df2.columns[0],
                            target_col=tgt_col,
                            freq=freq,
                        )

                    with st.expander("Validation report"):
                        st.json(validation_results)

                    results = []
                    model_labels = []

                    for ModelClass, label, kwargs in [
                        (ARIMA, "ARIMA", {"verbose": verbose}),
                        (SARIMA, "SARIMA", {"verbose": verbose}),
                        (LSTM, "LSTM", {"fut_pred": future_pred, "train_window": 10, "verbose": verbose}),
                    ]:
                        with st.spinner(f"Training {label}..."):
                            try:
                                if label == "LSTM":
                                    m = ModelClass(processed_data, feature_column=tgt_col, **kwargs)
                                else:
                                    m = ModelClass(processed_data, tgt_col, **kwargs)
                                perf, data_obj, model_obj = m.run()
                                results.append((perf, data_obj, model_obj))
                                model_labels.append(label)
                            except Exception as exc:
                                st.warning(f"{label} failed: {exc}")

                    if not results:
                        st.error("All models failed. Check the data and column selection.")
                    else:
                        perf_rows = [r[0]["performance"] for r in results]
                        perf_df = pd.DataFrame(perf_rows)

                        st.write("### Model comparison")
                        st.dataframe(perf_df)

                        best_model_name = Ranker(
                            df=pd.DataFrame(results_dicts := [r[0] for r in results]),
                            verbose=False,
                        ).get_best()
                        st.success(f"Best model: **{best_model_name}**")

                        # Visualize best model
                        best_idx = [r[0]["performance"]["Model"] for r in results].index(best_model_name)
                        _, best_data_obj, best_model_obj = results[best_idx]

                        if best_model_name in ("ARIMA", "SARIMA"):
                            with st.spinner("Generating visualization..."):
                                plt.close("all")
                                try:
                                    visualize_model(best_model_obj, best_data_obj, target_col=tgt_col)
                                    for fig_num in plt.get_fignums():
                                        st.pyplot(plt.figure(fig_num))
                                    plt.close("all")
                                except Exception as exc:
                                    st.info(f"Visualization skipped: {exc}")

                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                finally:
                    _os.chdir(original_cwd)
