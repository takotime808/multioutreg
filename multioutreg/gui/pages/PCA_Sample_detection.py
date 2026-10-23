# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows
import numpy as np
import pandas as pd
import streamlit as st

from multioutreg.sampling.detect import guess_input_columns
from multioutreg.sampling.pca_sampling_detector import generate_sampling_plot_and_metrics

st.title("Sampling Method Detection (PCA view)")

with st.sidebar:
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        st.warning("No numeric data found in file.")
    else:
        input_cols = st.sidebar.multiselect(
            "Input (design) columns",
            numeric_df.columns.tolist(),
            default=guess_input_columns(numeric_df.columns.tolist()),
            help="Responses are not part of the design; including them hides the sampling method.",
        )
        colour_by = st.sidebar.selectbox("Color by column (optional)", [None] + df.columns.tolist())
        if not input_cols:
            st.warning("Select at least one input column.")
        else:
            labels = df.loc[numeric_df.index, colour_by].to_numpy() if colour_by else None
            fig, metrics = generate_sampling_plot_and_metrics(
                numeric_df[input_cols].to_numpy(), random_state=int(random_state), labels=labels
            )
            st.subheader(f"Inferred sampling: {metrics['method']}")
            st.caption(metrics["explanation"])

            tab1, tab2 = st.tabs(["📊 PCA Plot", "📈 Metrics"])
            with tab1:
                st.pyplot(fig)
            with tab2:
                st.json(metrics)
else:
    st.info("Please upload a CSV file to begin analysis.")
