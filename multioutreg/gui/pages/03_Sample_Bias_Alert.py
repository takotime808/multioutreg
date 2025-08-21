# Copyright (c) 2025 takotime808

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_bias_metrics(df: pd.DataFrame) -> dict:
    """
    Compute basic metrics to detect potential sample bias in a dataset.

    This includes:
    - Duplicate row rate
    - Maximum absolute correlation between numeric features
    - Minimum and maximum variance across features

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing experimental or modeling data.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - "duplicate_rate": float
        - "max_abs_corr": float
        - "min_variance": float
        - "max_variance": float
    """
    metrics = {}
    numeric_df = df.select_dtypes(include=[np.number])
    metrics["duplicate_rate"] = df.duplicated().mean()
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr().abs()
        corr.values[[np.arange(len(corr))]*2] = 0  # zero self correlations
        metrics["max_abs_corr"] = corr.max().max()
    else:
        metrics["max_abs_corr"] = 0.0
    variances = numeric_df.var()
    metrics["min_variance"] = variances.min() if not variances.empty else 0.0
    metrics["max_variance"] = variances.max() if not variances.empty else 0.0
    return metrics


def scan_logs(log_path: str) -> list[str]:
    """
    Scan a log file for numerical stability issues such as Cholesky decomposition errors.

    Parameters
    ----------
    log_path : str
        Path to the log file to be scanned.

    Returns
    -------
    list of str
        A list of warning strings if Cholesky-related issues are found, otherwise an empty list.
    """
    alerts = []
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", errors="ignore") as f:
            text = f.read().lower()
        if re.search(r"cholesky|not positive definite", text):
            alerts.append("Cholesky decomposition issue found in logs")
    return alerts


def pca_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Generate a 2D PCA scatter plot from numeric columns in the input DataFrame.

    The data is first standardized using `StandardScaler`, then projected onto
    the first two principal components.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Only numeric columns are used, and rows with NaNs are dropped.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib figure containing the PCA scatter plot.
    """
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection")
    return fig


def main() -> None:
    st.title("Sample Bias Detection")
    uploaded = st.file_uploader("Upload DOE CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Data Preview", df.head())
        metrics = compute_bias_metrics(df)
        st.subheader("Bias Metrics")
        st.table(pd.DataFrame([metrics]))

        alerts = []
        if metrics["duplicate_rate"] > 0.05:
            alerts.append("High duplicate rate")
        if metrics["max_abs_corr"] > 0.95:
            alerts.append("Highly correlated features")
        if df.shape[1] > 3:
            st.subheader("PCA Visualization")
            fig = pca_plot(df)
            st.pyplot(fig)

        log_path = st.text_input("Log file to scan", value="output.log")
        alerts.extend(scan_logs(log_path))

        if alerts:
            st.error("\n".join(alerts))
        else:
            st.success("No significant bias detected")
    else:
        st.info("Please upload a CSV file to begin analysis")


if __name__ == "__main__":
    main()