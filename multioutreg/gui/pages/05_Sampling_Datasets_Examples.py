# Copyright (c) 2025 takotime808
"""Streamlit page for downloading example datasets made from different sampling methods."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc

from multioutreg.sampling.infer_sampling import infer_sampling_and_plot_umap

st.set_page_config(page_title="Sampling Datasets", layout="centered")

st.title("🧾 Downloadable Datasets Using Different Sampling Methods")

st.markdown(
    """
This page provides example datasets generated using different **sampling strategies**, useful for testing how various surrogate models respond to input distribution characteristics.
"""
)


def generate_sparse_alpha_cfd_dataset(
    selected_alphas: list = [0, 5, 10, 15, 20],
    beta_range: tuple = (-5, 5),
    reynolds_range: tuple = (1e5, 1e7),
    num_beta: int = 5,
    num_re: int = 5,
    noise_std: float = 0.01,
) -> pd.DataFrame:
    beta_vals = np.linspace(*beta_range, num_beta)
    reynolds_vals = np.logspace(np.log10(reynolds_range[0]), np.log10(reynolds_range[1]), num_re)
    alpha_vals = np.array(selected_alphas)

    alpha_grid, beta_grid, re_grid = np.meshgrid(alpha_vals, beta_vals, reynolds_vals, indexing="ij")
    alpha = alpha_grid.ravel()
    beta = beta_grid.ravel()
    Re = re_grid.ravel()

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)

    Cl = 2 * np.pi * alpha_rad * np.cos(beta_rad)
    Cl[Cl > 1.5] = 1.5 - 5 * (Cl[Cl > 1.5] - 1.5) ** 2
    Cl += 0.1 * (np.log10(Re / 1e6))
    Cl += np.random.normal(0, noise_std, size=Cl.shape)

    df = pd.DataFrame({
        "alpha_deg": alpha,
        "beta_deg": beta,
        "Re": Re,
        "Cl": Cl,
    })

    return df


def generate_grid_sampling_dataset(num_points_per_axis: int = 10):
    x = np.linspace(0, 1, num_points_per_axis)
    xx, yy = np.meshgrid(x, x)
    x1 = xx.ravel()
    x2 = yy.ravel()
    df = pd.DataFrame({"x1": x1, "x2": x2})
    df["response"] = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2)
    return "Grid Sampled Dataset", df


def generate_random_sampling_dataset(n_points: int = 100):
    rng = np.random.default_rng(seed=42)
    X = rng.uniform(0, 1, size=(n_points, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Randomly Sampled Dataset", df


def generate_sobol_sampling_dataset(n_points: int = 128):
    sampler = qmc.Sobol(d=2, scramble=False, seed=42)
    X = sampler.random_base2(m=int(np.log2(n_points)))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Sobol Sampled Dataset", df


def generate_lhs_sampling_dataset(n_points: int = 100):
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


def fix_generate_grid_sampling_dataset():
    df = generate_sparse_alpha_cfd_dataset(num_beta=30, num_re=30, noise_std=0)
    return "Grid Sampled Dataset", df


datasets = [
    generate_lhs_sampling_dataset(),
    generate_grid_sampling_dataset(),
    generate_random_sampling_dataset(),
    generate_sobol_sampling_dataset(),
    generate_uncertain_sampling_dataset(),
    fix_generate_grid_sampling_dataset(),
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
                inferred_method, fig, explanation = infer_sampling_and_plot_umap(X, explanation_indicator=True)
                st.pyplot(fig)
                if "lhs" in name.lower():
                    expected_method = "LHS"
                elif "grid" in name.lower():
                    expected_method = "Grid"
                elif "sobol" in name.lower():
                    expected_method = "Sobol"
                elif "random" in name.lower():
                    expected_method = "Random"
                else:
                    expected_method = "Unknown"
                st.markdown(f"""\
🏷️ **Expected Method:** `{expected_method}`  
🔍 **Inferred Method:** `{inferred_method}`  
🧠 **Explanation:** {explanation}
""")
            except Exception as e:
                st.warning(f"Could not analyze UMAP: {e}")
        else:
            st.warning("Not enough numeric columns for UMAP analysis.")

    st.markdown("---")