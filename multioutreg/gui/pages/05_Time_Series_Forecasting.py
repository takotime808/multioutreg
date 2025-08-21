# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from multioutreg.time_series.chronos_adapter import ChronosForecaster

st.set_page_config(page_title="Time-Series Forecasting (Chronos)", layout="wide")

st.title("Time-Series Forecasting (Chronos / Chronos-Bolt)")
st.caption("Zero-shot probabilistic forecasting using Chronos-style foundation models.")

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
model = st.selectbox("Model", ["amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small", "amazon/chronos-bolt-base", "amazon/chronos-t5-small"])
horizon = st.number_input("Prediction horizon (steps)", min_value=1, value=24)
q_text = st.text_input("Quantiles (comma-separated)", "0.1,0.5,0.9")

if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())
    time_col = st.selectbox("Time column (optional)", ["<none>"] + list(df.columns))
    numeric = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    targets = st.multiselect("Target columns", numeric, default=(numeric[:1] if numeric else []))
    go = st.button("Run forecast")

    if go and targets:
        forecaster = ChronosForecaster(model_name=model)
        series = {c: df[c].dropna().to_numpy() for c in targets}
        res = forecaster.fit(series).predict(horizon, quantiles=[float(x) for x in q_text.split(",") if x.strip()])

        # plot
        for i, sid in enumerate(res.ids):
            hist = series[sid]
            fig = plt.figure()
            plt.plot(np.arange(len(hist)), hist, label="history")
            for qi, q in enumerate(res.q_levels):
                plt.plot(np.arange(len(hist), len(hist)+horizon), res.quantiles[i, qi, :], label=f"q{q}")
            plt.title(sid)
            plt.legend()
            st.pyplot(fig)
