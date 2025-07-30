# Copyright (c) 2025 takotime808

"""Streamlit page for downloading example datasets made from different sampling methods."""

import numpy as np
import pandas as pd
# import seaborn as sns
import streamlit as st
from scipy.stats import qmc
# import matplotlib.pyplot as plt

from multioutreg.sampling.infer_sampling import infer_sampling_and_plot_umap

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


def generate_sparse_alpha_cfd_dataset(
    selected_alphas: list = [0, 5, 10, 15, 20],
    beta_range: tuple = (-5, 5),
    reynolds_range: tuple = (1e5, 1e7),
    num_beta: int = 5,
    num_re: int = 5,
    noise_std: float = 0.01,
    include_metadata: bool = False
) -> pd.DataFrame:
    """
    Generate a synthetic CFD dataset for selected alpha values only.

    Parameters
    ----------
    selected_alphas : list
        List of alpha values (in degrees) to include.
    beta_range : tuple
        Range of sideslip angle beta in degrees.
    reynolds_range : tuple
        Range of Reynolds number (min, max).
    num_beta : int
        Number of beta samples.
    num_re : int
        Number of Reynolds number samples.
    noise_std : float
        Standard deviation of added Gaussian noise.
    include_metadata : bool
        If True, return a DataFrame with all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: alpha_deg, beta_deg, Re, Cl, Cd
    """
    # Meshgrid for beta and Re
    beta_vals = np.linspace(*beta_range, num_beta)
    reynolds_vals = np.logspace(np.log10(reynolds_range[0]), np.log10(reynolds_range[1]), num_re)
    alpha_vals = np.array(selected_alphas)

    # Create full meshgrid
    alpha_grid, beta_grid, re_grid = np.meshgrid(alpha_vals, beta_vals, reynolds_vals, indexing='ij')
    alpha = alpha_grid.ravel()
    beta = beta_grid.ravel()
    Re = re_grid.ravel()

    # Convert to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)

    # Generate Lift Coefficient (Cl)
    Cl = 2 * np.pi * alpha_rad * np.cos(beta_rad)
    Cl[Cl > 1.5] = 1.5 - 5 * (Cl[Cl > 1.5] - 1.5)**2  # stall effect
    Cl += 0.1 * (np.log10(Re / 1e6))
    Cl += np.random.normal(0, noise_std, size=Cl.shape)

    # Generate Drag Coefficient (Cd)
    Cd = 0.01 + 0.02 * alpha_rad**2 + 0.01 * beta_rad**2
    Cd += 0.02 * (np.log10(Re / 1e6))**2
    Cd += np.random.normal(0, noise_std / 2, size=Cd.shape)

    df = pd.DataFrame({
        "alpha_deg": alpha,
        "beta_deg": beta,
        "Re": Re,
        "Cl": Cl,
        # "Cd": Cd
    })

    return df


def fix_generate_grid_sampling_dataset():
    """
    Generate a gridded regression dataset using a smooth function over 2D space.

    This example uses the function:
        f(x, y) = sin(x) + cos(y) + (x * y) / 10

    Returns
    -------
    Tuple[str, pd.DataFrame]
        Dataset name and DataFrame with columns: ["x1", "x2", "response"]
    """

    # Generate the DataFrame
    df = generate_sparse_alpha_cfd_dataset(num_beta=30,num_re=30,noise_std=0)

    return "Grid Sampled Dataset", df


def generate_grid_sampling_dataset(num_points_per_axis: int = 10):
    """
    Generate a structured grid-sampled dataset with 5 input features and a nonlinear response.

    Parameters
    ----------
    num_points_per_axis : int, optional
        Number of grid points per axis for x1 and x2 (default is 10).

    Returns
    -------
    Tuple[str, pd.DataFrame]
        Dataset name and DataFrame with columns ["x1", ..., "x5", "response"]
    """
    # Create structured grid in 2D space
    x = np.linspace(0, 1, num_points_per_axis)
    xx, yy = np.meshgrid(x, x)
    x1 = xx.ravel()
    x2 = yy.ravel()

    # Extend to 5D by creating engineered features
    x3 = (x1 + x2) / 2
    x4 = np.sin(np.pi * x1) * np.cos(np.pi * x2)
    x5 = x1**2 + x2**2

    # Stack features
    X = np.column_stack([x1, x2, x3, x4, x5])

    # Nonlinear response
    response = np.sin(2 * np.pi * x1) + np.cos(2 * np.pi * x2) + 0.5 * x3 - 0.3 * x5

    # Construct DataFrame
    df = pd.DataFrame(X, columns=["x1", "x2", "x3", "x4", "x5"])
    df["response"] = response

    return "Grid Sampled Dataset", df


def generate_random_sampling_dataset(n_points=100):
    """
    Generate a dataset using random uniform sampling.
    Designed to produce high std and low silhouette → Random.

    Returns
    -------
    Tuple[str, pd.DataFrame]
    """
    rng = np.random.default_rng(seed=42)
    X = rng.uniform(0, 1, size=(n_points, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Randomly Sampled Dataset", df


def generate_sobol_sampling_dataset(n_points=128):
    """
    Generate a dataset using Sobol low-discrepancy sequence.
    Designed to produce moderate std and low entropy → Sobol.

    Returns
    -------
    Tuple[str, pd.DataFrame]
    """
    sampler = qmc.Sobol(d=2, scramble=False, seed=42)
    X = sampler.random_base2(m=int(np.log2(n_points)))
    X = qmc.scale(X, 0, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "Sobol Sampled Dataset", df


def generate_lhs_sampling_dataset(n_points=100):
    """
    Generate a dataset using Latin Hypercube Sampling.
    Designed to produce moderate std and higher entropy → LHS.

    Returns
    -------
    Tuple[str, pd.DataFrame]
    """
    sampler = qmc.LatinHypercube(d=2, seed=42)
    X = sampler.random(n=n_points)
    X = qmc.scale(X, 0, 1)
    y = np.sin(2 * np.pi * X[:, 0]) + np.cos(2 * np.pi * X[:, 1])
    df = pd.DataFrame(np.column_stack([X, y]), columns=["x1", "x2", "response"])
    return "LHS Sampled Dataset", df


def generate_uncertain_sampling_dataset():
    """
    Generate an uncertain sampling dataset with fixed values.
    Intended to strongly trigger 'Uncertain' detection.

    Returns
    -------
    Tuple[str, pd.DataFrame]
        Dataset name and DataFrame.
    """
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
    fix_generate_grid_sampling_dataset(),
]


# for name, df in datasets:
#     st.subheader(f"📊 {name}")
#     col1, col2 = st.columns([3, 2])  # 3:2 width ratio

#     with col1:
#         st.dataframe(df.head(), use_container_width=True)

#         csv_bytes = df.to_csv(index=False).encode("utf-8")
#         filename = name.lower().replace(" ", "_").replace("-", "_") + ".csv"

#         st.download_button(
#             label="⬇️ Download CSV",
#             data=csv_bytes,
#             file_name=filename,
#             mime="text/csv",
#         )

#     # with col2:
#     #     try:
#     #         numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#     #         if len(numeric_cols) >= 2:
#     #             g = sns.PairGrid(df[numeric_cols], diag_sharey=False)
#     #             g.map_upper(sns.scatterplot, s=10, alpha=0.6)
#     #             g.map_lower(sns.kdeplot, fill=True, thresh=0.05, cmap="Blues")
#     #             g.map_diag(sns.histplot, kde=True)

#     #             g.fig.set_size_inches(8, 6)
#     #             st.pyplot(g.fig)
#     #             plt.close(g.fig)
#     #         else:
#     #             st.info("Not enough numeric columns for visualization.")
#     #     except Exception as e:
#     #         st.warning(f"Could not generate pairplot: {e}")
#     with col2:
#         try:
#             numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#             if len(numeric_cols) >= 2:
#                 # PairGrid
#                 g = sns.PairGrid(df[numeric_cols], diag_sharey=False)
#                 g.map_upper(sns.scatterplot, s=10, alpha=0.6)
#                 g.map_lower(sns.kdeplot, fill=True, thresh=0.05, cmap="Blues")
#                 g.map_diag(sns.histplot, kde=True)
#                 g.fig.set_size_inches(6, 4)
#                 st.pyplot(g.fig)
#                 plt.close(g.fig)

#                 # UMAP sampling detection
#                 method, fig_umap = detect_sampling_method_and_plot(df[numeric_cols].values)
#                 st.pyplot(fig_umap)
#                 st.caption(f"🔍 Sampling Detection: **{method}**")
#             else:
#                 st.info("Not enough numeric columns for visualization.")
#         except Exception as e:
#             st.warning(f"Could not generate pairplot: {e}")

#     st.markdown("---")


# Loop through all datasets
for name, df in datasets:
    st.subheader(f"📊 {name}")
    col1, col2 = st.columns([3, 2])  # Data | Visual

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

                # Ground truth method (from name)
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

                # Show comparison
                st.markdown(
                    f"""
                    🏷️ **Expected Method:** `{expected_method}`  
                    🔍 **Inferred Method:** `{inferred_method}`  
                    🧠 **Explanation:** {explanation}
                    """
                )
            except Exception as e:
                st.warning(f"Could not analyze UMAP: {e}")
        else:
            st.warning("Not enough numeric columns for UMAP analysis.")

    st.markdown("---")

