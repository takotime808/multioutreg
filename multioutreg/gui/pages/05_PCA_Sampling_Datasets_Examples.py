# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc
from multioutreg.sampling.pca_sampling_detector import generate_sampling_plot_and_metrics

st.set_page_config(page_title="Sampling Datasets", layout="centered")

st.title("🧾 Downloadable Datasets Using Different Sampling Methods")

st.markdown(
    """
This page provides example datasets generated using different **sampling strategies**, useful for testing how various surrogate models respond to input distribution characteristics.

Each dataset demonstrates a specific method:

- 🧮 **Grid Sampling** — Regular, low-variance sampling  
- 🎲 **Random Sampling** — Uniform random draws from the input space  
- 📈 **Latin Hypercube Sampling (LHS)** — Stratified sampling with uniform marginals  
- 🔁 **Sobol Sequence** — Low-discrepancy quasi-random sampling  

These datasets can be used to test how sampling bias affects prediction accuracy, uncertainty calibration, and generalizability.
"""
)

def generate_grid_sampling_dataset(num_points_per_axis: int = 10):
    x = np.linspace(0, 1, num_points_per_axis)
    xx, yy = np.meshgrid(x, x)
    x1 = xx.ravel()
    x2 = yy.ravel()
    x3 = (x1 + x2) / 2
    x4 = np.sin(np.pi * x1) * np.cos(np.pi * x2)
    x5 = x1**2 + x2**2
    X = np.column_stack([x1, x2, x3, x4, x5])
    response = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + 0.5 * x3 - 0.3 * x5
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    df["response"] = response
    return "Grid Sampled Dataset", df

def generate_random_sampling_dataset(n_points=100):
    rng = np.random.default_rng(seed=42)
    X = rng.uniform(0, 1, size=(n_points, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Randomly Sampled Dataset", df

def generate_sobol_sampling_dataset(n_points=128):
    sampler = qmc.Sobol(d=2, scramble=False, seed=42)
    X = sampler.random_base2(m=int(np.log2(n_points)))
    X = qmc.scale(X, 0, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Sobol Sampled Dataset", df

def generate_lhs_sampling_dataset(n_points=100):
    sampler = qmc.LatinHypercube(d=2, seed=42)
    X = sampler.random(n=n_points)
    X = qmc.scale(X, 0, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "LHS Sampled Dataset", df

def generate_uncertain_sampling_dataset():
    df = pd.DataFrame({
        "x1": [0, 1, 2, 3, 4, 5],
        "x2": [0, 10, 20, 25, 30, 35],
        "response": [0, 1, 2, 4, 5, 6],
    })
    return "'Uncertain' Sampled Dataset", df

# Register datasets
datasets = [
    generate_lhs_sampling_dataset(),
    generate_grid_sampling_dataset(),
    generate_random_sampling_dataset(),
    generate_sobol_sampling_dataset(),
    generate_uncertain_sampling_dataset(),
]

for name, df in datasets:
    st.subheader(f"📊 {name}")
    col1, col2 = st.columns([3, 2])

    with col1:
        st.dataframe(df.head(), use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = name.lower().replace(" ", "_").replace("-", "_") + ".csv"
        st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=filename, mime="text/csv")

    with col2:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            X = df[numeric_cols].values
            try:
                fig, metrics = generate_sampling_plot_and_metrics(X, random_state=42)
                st.pyplot(fig)

                st.markdown(
                    f"""
                    🏷️ **Expected Method:** `{name.split()[0]}`  
                    🔍 **Inferred Method:** `{metrics['method']}`  
                    🧠 **Explanation:** {metrics['explanation']}
                    """
                )
            except Exception as e:
                st.warning(f"Could not analyze PCA: {e}")
        else:
            st.warning("Not enough numeric columns for PCA analysis.")

    st.markdown("---")