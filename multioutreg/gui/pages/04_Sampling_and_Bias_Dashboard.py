# Copyright (c) 2025 takotime808

import os
import re
import tempfile
import base64
import numpy as np
import pandas as pd
import streamlit as st
from typer.testing import CliRunner
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mor_cli.detect_sampling import app as detect_app
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.pca_plots import generate_pca_variance_plot


def compute_bias_metrics(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    metrics = {
        "duplicate_rate": df.duplicated().mean(),
        "max_abs_corr": numeric_df.corr().abs().where(
            ~np.eye(numeric_df.shape[1], dtype=bool)).max().max() if numeric_df.shape[1] > 1 else 0.0,
        "min_variance": numeric_df.var().min() if not numeric_df.empty else 0.0,
        "max_variance": numeric_df.var().max() if not numeric_df.empty else 0.0
    }
    return metrics


def scan_logs(log_path: str) -> list[str]:
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", errors="ignore") as f:
            if re.search(r"cholesky|not positive definite", f.read().lower()):
                return ["Cholesky decomposition issue found in logs"]
    return []


def show_base64_image(b64_str: str, caption: str = ""):
    if b64_str:
        st.image(f"data:image/png;base64,{b64_str}", caption=caption, use_column_width=True)


def main():
    st.title("🧪 Sampling & Bias Dashboard")

    uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        column_options = df.columns.tolist()

        # Sidebar Controls
        st.sidebar.title("🔧 Visualization Parameters")
        st.sidebar.subheader("UMAP")
        umap_neighbors = st.sidebar.slider("UMAP: n_neighbors", 5, 50, 15)
        umap_min_dist = st.sidebar.slider("UMAP: min_dist", 0.0, 1.0, 0.1)
        umap_target = st.sidebar.selectbox("Color by column (optional)", [None] + column_options)

        st.sidebar.subheader("PCA")
        pca_thresh = st.sidebar.slider("PCA: Variance Threshold (optional)", 0.0, 1.0, 0.9)

        # Tabs
        tabs = st.tabs(["📊 Dataset & Sampling", "🚨 Bias Metrics", "📈 PCA", "🔍 UMAP"])

        # Tab 1: Dataset & Sampling
        with tabs[0]:
            st.subheader("🔍 Dataset Preview")
            st.write(df.head())

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                df.to_csv(tmp.name, index=False)
                runner = CliRunner()
                result = runner.invoke(detect_app, [tmp.name])
                st.subheader("📌 Detected Sampling Method")
                if result.exit_code == 0:
                    st.text(result.stdout)
                else:
                    st.error(result.stdout or result.stderr)

        # Tab 2: Bias Metrics
        with tabs[1]:
            st.subheader("📉 Bias Metrics")
            metrics = compute_bias_metrics(df)
            st.table(pd.DataFrame([metrics]))

            alerts = []
            if metrics["duplicate_rate"] > 0.05:
                alerts.append("⚠️ High duplicate rate")
            if metrics["max_abs_corr"] > 0.95:
                alerts.append("⚠️ Highly correlated features")

            log_path = st.text_input("🔍 Log file to scan (optional)", value="output.log")
            alerts.extend(scan_logs(log_path))

            if alerts:
                st.error("🚨 Alerts:\n" + "\n".join(alerts))
            else:
                st.success("✅ No significant bias detected")

        # Tab 3: PCA
        with tabs[2]:
            if numeric_df.shape[1] >= 2:
                st.subheader("📈 PCA Scree Plot")
                pca = PCA().fit(StandardScaler().fit_transform(numeric_df))
                pca_b64 = generate_pca_variance_plot(pca, threshold=pca_thresh)
                show_base64_image(pca_b64, caption="Explained Variance per Component")
            else:
                st.warning("⚠️ Not enough numeric features for PCA.")

        # Tab 4: UMAP
        with tabs[3]:
            if numeric_df.shape[1] >= 2:
                st.subheader("🔍 UMAP Projection")
                label_data = df[umap_target].to_numpy() if umap_target else None
                umap_b64, umap_explanation = generate_umap_plot(
                    X=numeric_df.to_numpy(),
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    labels=label_data
                )
                show_base64_image(umap_b64, caption=f"UMAP Projection — {umap_explanation}")
            else:
                st.warning("⚠️ Not enough numeric features for UMAP.")
    else:
        st.info("📥 Please upload a CSV file to begin analysis.")


if __name__ == "__main__":
    main()
