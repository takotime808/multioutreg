# Copyright (c) 2025 takotime808

"""Streamlit page for downloading example datasets with and without fidelity levels."""

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Example Multi-Output Datasets", layout="centered")

st.title("🧾 Downloadable Multi-Output Example Datasets")

st.markdown(
    """
This page provides example datasets for exploring multi-output surrogate modeling,
including support for uncertainty quantification, multiple fidelities, and real-world design of experiments (DOE) data.

Each dataset demonstrates different characteristics:

- 🧪 Linear and nonlinear relationships  
- 🎯 Single and multi-output targets  
- 🧬 Multi-fidelity simulation levels (with numeric fidelity indicators)  
- 🕒 Timestamped samples and distribution drift  
- ⚖️ Class imbalance in labeled DOE

You can download these datasets and upload them to other pages in the app to test model training and reporting features.
"""
)

# Dataset 1: Simple Linear Model
def generate_dataset_linear():
    """Multi-output linear dataset."""
    np.random.seed(0)
    X = np.random.rand(100, 3)
    y1 = np.dot(X, [1.5, -2.0, 1.0]) + 0.1 * np.random.randn(100)
    y2 = np.dot(X, [-1.0, 0.5, 2.0]) + 0.1 * np.random.randn(100)
    df = pd.DataFrame(np.hstack([X, y1[:, None], y2[:, None]]), columns=["x1", "x2", "x3", "y1", "y2"])
    return "Simple Linear Multi-Output Dataset", df


# Dataset 2: Nonlinear Multi-Output
def generate_dataset_nonlinear():
    """Multi-output non-linear dataset."""
    np.random.seed(1)
    X = np.random.rand(120, 4)
    y1 = np.sin(X[:, 0]) + X[:, 1]**2
    y2 = np.log1p(X[:, 2]) - X[:, 3]
    df = pd.DataFrame(np.column_stack([X, y1, y2]), columns=["f1", "f2", "f3", "f4", "output1", "output2"])
    return "Nonlinear Multi-Output Dataset", df


# Dataset 3: Multi-Fidelity Example
def generate_dataset_multifidelity():
    """Multi-output multi-fidelity dataset."""
    np.random.seed(42)
    n_per_fidelity = 100
    fidelities = np.array([0] * n_per_fidelity + [1] * n_per_fidelity + [2] * n_per_fidelity)
    X = np.random.rand(n_per_fidelity * 3, 2)

    y1, y2 = [], []
    for i, f in enumerate(fidelities):
        x1, x2 = X[i]
        if f == 0:
            y1.append(np.sin(2 * np.pi * x1) + 0.5 * np.random.randn())
            y2.append(np.cos(2 * np.pi * x2) + 0.5 * np.random.randn())
        elif f == 1:
            y1.append(np.sin(2 * np.pi * x1) + 0.2 * np.random.randn())
            y2.append(np.cos(2 * np.pi * x2) + 0.2 * np.random.randn())
        else:
            y1.append(np.sin(2 * np.pi * x1))
            y2.append(np.cos(2 * np.pi * x2))

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["fidelity_level"] = fidelities
    df["y1"] = y1
    df["y2"] = y2
    return "Multi-Fidelity Multi-Output Dataset (numeric fidelity level)", df


# Dataset 4: DOE Dataset with Class Imbalance and Timestamps
def generate_dataset_doe_with_class_imbalance():
    """Multi-output DOE with class imbalance and timestamps."""
    np.random.seed(123)
    n_samples = 200
    X = np.random.uniform(0, 1, size=(n_samples, 3))
    classes = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    base_time = pd.Timestamp("2023-01-01")
    timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n_samples)]

    y1 = X[:, 0] * 2 + np.sin(X[:, 1] * np.pi) + classes * 1.5 + np.random.normal(0, 0.1, n_samples)
    y2 = np.cos(np.pi * X[:, 2]) + classes + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame(X, columns=["temperature", "pressure", "speed"])
    df["class_label"] = classes
    df["timestamp"] = timestamps
    df["response1"] = y1
    df["response2"] = y2
    return "DOE Dataset (Class Imbalance + Timestamps, Multi-Output)", df


# Dataset 5: DOE Dataset with drift
def generate_dataset_doe_with_drift():
    """Multi-output DOE with drift, imbalance, and timestamps."""
    np.random.seed(123)
    n_samples = 200
    X = np.random.uniform(0, 1, size=(n_samples, 3))
    classes = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    base_time = pd.Timestamp("2023-01-01")
    timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(n_samples)]
    drift_weights = np.linspace(2.0, 5.0, n_samples)

    temperature = X[:, 0]
    pressure = X[:, 1]
    speed = X[:, 2]

    y1 = drift_weights * temperature + np.sin(np.pi * pressure) + classes * 1.5 + np.random.normal(0, 0.1, n_samples)
    y2 = np.cos(np.pi * speed) + drift_weights * 0.5 + np.random.normal(0, 0.1, n_samples)

    df = pd.DataFrame(X, columns=["temperature", "pressure", "speed"])
    df["class_label"] = classes
    df["timestamp"] = timestamps
    df["response1"] = y1
    df["response2"] = y2
    return "DOE Dataset (Class Imbalance + Timestamps + Drift, Multi-Output)", df


# Register datasets
datasets = [
    generate_dataset_linear(),
    generate_dataset_nonlinear(),
    generate_dataset_multifidelity(),
    generate_dataset_doe_with_class_imbalance(),
    generate_dataset_doe_with_drift(),
]


for name, df in datasets:
    st.subheader(f"📊 {name}")
    col1, col2 = st.columns([3, 2])  # 3:2 width ratio

    with col1:
        st.dataframe(df.head(), use_container_width=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = name.lower().replace(" ", "_").replace("-", "_") + ".csv"

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )

    with col2:
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                g = sns.PairGrid(df[numeric_cols], diag_sharey=False)
                g.map_upper(sns.scatterplot, s=10, alpha=0.6)
                g.map_lower(sns.kdeplot, fill=True, thresh=0.05, cmap="Blues")
                g.map_diag(sns.histplot, kde=True)

                g.fig.set_size_inches(8, 6)
                st.pyplot(g.fig)
                plt.close(g.fig)
            else:
                st.info("Not enough numeric columns for visualization.")
        except Exception as e:
            st.warning(f"Could not generate pairplot: {e}")

    st.markdown("---")
