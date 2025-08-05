# Copyright (c) 2025 takotime808

import os
import re
import tempfile
import base64
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
import seaborn as sns
import matplotlib.pyplot as plt

from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.pca_plots import generate_pca_variance_plot


def plot_pairgrid_with_kde(df, numeric_cols):
    st.subheader("Pairwise Relationships with Seaborn PairGrid")

    if len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for pairwise visualization.")
        return

    g = sns.PairGrid(df[numeric_cols])

    # Upper triangle: scatter plots
    g.map_upper(sns.scatterplot, s=15, edgecolor="k", alpha=0.7)

    # Lower triangle: KDE plots
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True, thresh=0.05)

    # Diagonal: histograms
    g.map_diag(sns.histplot, edgecolor="black", bins=15)

    plt.tight_layout()
    st.pyplot(g.fig)


# --- LHS/Grid Detection Utilities ---

def compute_discrepancy(X):
    N, d = X.shape
    disc = 0
    for i in range(N):
        prod1 = np.prod(1 - X[i] ** 2)
        prod2 = np.prod(0.5 - np.abs(0.5 - X[i]))
        disc += prod1 - prod2
    return (13.0 / 12.0) ** d - (2.0 / N) * disc

def compute_maximin_distance(X):
    return np.min(pdist(X))

def compare_with_random(X, n_trials=10):
    N, d = X.shape
    lhs_disc = compute_discrepancy(X)
    lhs_maximin = compute_maximin_distance(X)
    rand_discs = []
    rand_maxs = []
    for _ in range(n_trials):
        X_rand = np.random.rand(N, d)
        rand_discs.append(compute_discrepancy(X_rand))
        rand_maxs.append(compute_maximin_distance(X_rand))
    return {
        "lhs_discrepancy": lhs_disc,
        "lhs_maximin": lhs_maximin,
        "random_discrepancy_mean": np.mean(rand_discs),
        "random_maximin_mean": np.mean(rand_maxs)
    }

def determine_sampling_type(X_scaled: np.ndarray, corr_df: pd.DataFrame):
    metrics = compare_with_random(X_scaled, n_trials=10)
    mean_corr = np.mean(np.abs(corr_df.values[np.triu_indices_from(corr_df, 1)]))
    lhs_like = (
        metrics["lhs_discrepancy"] < 0.9 * metrics["random_discrepancy_mean"] and
        metrics["lhs_maximin"] > 1.1 * metrics["random_maximin_mean"] and
        mean_corr < 0.2
    )
    return lhs_like, mean_corr, metrics

def is_grid_sample(X: np.ndarray, tol=1e-6) -> bool:
    n, d = X.shape
    unique_axes = [np.unique(X[:, i]) for i in range(d)]
    sizes = [len(u) for u in unique_axes]
    if np.prod(sizes) != n:
        return False
    grid = np.stack(np.meshgrid(*unique_axes, indexing='ij'), -1).reshape(-1, d)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    dists, _ = tree.query(grid, k=1)
    return np.all(dists < tol)

# --- Bias + Viz Utilities ---

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

# --- Main App ---

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

        # --- Tab 1: Sampling Detection ---
        with tabs[0]:
            st.subheader("🔍 Dataset Preview")
            st.write(df.head())

            if numeric_df.shape[1] < 2:
                st.error("❌ Not enough numeric features for sampling detection.")
            else:
                # Normalize for analysis
                X_scaled = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
                X_scaled = X_scaled.fillna(0.5).to_numpy()
                corr_df = pd.DataFrame(X_scaled).corr(method="spearman")

                lhs_like, mean_corr, metrics = determine_sampling_type(X_scaled, corr_df)
                st.subheader("📌 Detected Sampling Method")

                st.markdown(f"""
                - **Discrepancy (lower is better)**: `{metrics['lhs_discrepancy']:.4f}` (random avg: `{metrics['random_discrepancy_mean']:.4f}`)
                - **Maximin distance (higher is better)**: `{metrics['lhs_maximin']:.4f}` (random avg: `{metrics['random_maximin_mean']:.4f}`)
                - **Mean absolute Spearman correlation**: `{mean_corr:.4f}`
                """)

                if lhs_like:
                    st.success("✅ Likely LHS-like sampling detected.")
                elif is_grid_sample(X_scaled):
                    st.warning("🟨 Likely structured grid sampling detected.")
                else:
                    st.error("❌ Sampling pattern appears random or unknown.")

                # New: Visual pairwise plot with PairGrid
                plot_pairgrid_with_kde(df, numeric_df.columns.tolist())


        # --- Tab 2: Bias Metrics ---
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

        # --- Tab 3: PCA ---
        with tabs[2]:
            if numeric_df.shape[1] >= 2:
                st.subheader("📈 PCA Scree Plot")
                pca = PCA().fit(StandardScaler().fit_transform(numeric_df))
                pca_b64 = generate_pca_variance_plot(pca, threshold=pca_thresh)
                show_base64_image(pca_b64, caption="Explained Variance per Component")
            else:
                st.warning("⚠️ Not enough numeric features for PCA.")

        # --- Tab 4: UMAP ---
        with tabs[3]:
            if numeric_df.shape[1] >= 2:
                st.subheader("🔍 UMAP Projection")
                umap_b64, umap_explanation = generate_umap_plot(
                    X=numeric_df.to_numpy(),
                    # Uncomment below if you want full control
                    # n_neighbors=umap_neighbors,
                    # min_dist=umap_min_dist,
                    # labels=df[umap_target].to_numpy() if umap_target else None
                )
                show_base64_image(umap_b64, caption=f"UMAP Projection — {umap_explanation}")
            else:
                st.warning("⚠️ Not enough numeric features for UMAP.")
    else:
        st.info("📥 Please upload a CSV file to begin analysis.")


if __name__ == "__main__":
    main()
