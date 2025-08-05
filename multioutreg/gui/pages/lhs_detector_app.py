# Copyright (c) 2025 takotime808

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# --------- Utility Functions ----------

def plot_pairwise(df):
    fig = sns.pairplot(df)
    st.pyplot(fig)

def plot_histograms(df):
    fig, axs = plt.subplots(nrows=len(df.columns), figsize=(8, 2 * len(df.columns)))
    for i, col in enumerate(df.columns):
        axs[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(f"Histogram of {col}")
    plt.tight_layout()
    st.pyplot(fig)

def plot_correlation_matrix(df):
    corr = df.corr(method='spearman')
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    return corr

def compute_discrepancy(X):
    N, d = X.shape
    disc = 0
    for i in range(N):
        prod1 = 1
        prod2 = 1
        for j in range(d):
            prod1 *= (1 - X[i, j] ** 2)
            prod2 *= (0.5 - abs(0.5 - X[i, j]))
        disc += prod1 - prod2
    return (13.0 / 12.0) ** d - (2.0 / N) * disc

def compute_maximin_distance(X):
    dists = pdist(X)
    return np.min(dists)

def compare_with_random(X, n_trials=10):
    N, d = X.shape
    lhs_disc = compute_discrepancy(X)
    lhs_maximin = compute_maximin_distance(X)

    random_discrepancies = []
    random_maximins = []
    for _ in range(n_trials):
        X_rand = np.random.rand(N, d)
        random_discrepancies.append(compute_discrepancy(X_rand))
        random_maximins.append(compute_maximin_distance(X_rand))

    return {
        "lhs_discrepancy": lhs_disc,
        "lhs_maximin": lhs_maximin,
        "random_discrepancy_mean": np.mean(random_discrepancies),
        "random_maximin_mean": np.mean(random_maximins),
    }

def determine_sampling_type(metrics, corr):
    mean_corr = np.mean(np.abs(corr.values[np.triu_indices_from(corr, 1)]))
    lhs_like = (
        metrics["lhs_discrepancy"] < metrics["random_discrepancy_mean"] * 0.9 and
        metrics["lhs_maximin"] > metrics["random_maximin_mean"] * 1.1 and
        mean_corr < 0.2
    )
    return lhs_like, mean_corr

def is_grid_sample(X: np.ndarray, tol=1e-6) -> bool:
    n_points, n_dims = X.shape
    unique_per_dim = [np.unique(X[:, i]) for i in range(n_dims)]
    sizes = [len(u) for u in unique_per_dim]
    expected_points = np.prod(sizes)

    # Check if actual number of points matches the product of unique levels
    if expected_points != n_points:
        return False

    # Check Cartesian product structure by regenerating grid and comparing
    mesh = np.meshgrid(*unique_per_dim, indexing='ij')
    grid_points = np.stack([m.flatten() for m in mesh], axis=1)

    # Compare sets of points with tolerance
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    distances, _ = tree.query(grid_points, k=1)
    return np.all(distances < tol)


# --------- Streamlit UI ----------

st.title("🧪 LHS Detection Tool")
st.write("""
This app helps you evaluate whether uploaded data might have been generated using **Latin Hypercube Sampling (LHS)**.
It combines visual inspection, correlation analysis, and uniformity metrics.
""")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.header("🔍 Data Summary and Preprocessing")
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if categorical_cols:
        st.warning(f"⚠️ Categorical columns excluded: {', '.join(categorical_cols)}")

    numeric_df = df[numeric_cols].copy()

    constant_cols = [col for col in numeric_df.columns if numeric_df[col].nunique() == 1]
    if constant_cols:
        st.warning(f"⚠️ Constant columns dropped: {', '.join(constant_cols)}")
        numeric_df = numeric_df.drop(columns=constant_cols)

    if numeric_df.empty:
        st.error("❌ No valid numeric columns left for analysis.")
        st.stop()

    # Normalize to [0, 1]
    scaled_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    scaled_df = scaled_df.fillna(0.5)  # fallback in case of zero division
    st.success("✅ Numeric data normalized to [0, 1] for LHS analysis.")

    # --- Visual Inspection ---
    st.subheader("1. 📊 Visual Inspection")
    st.write("### Pairwise Relationships")
    plot_pairwise(scaled_df)

    st.write("### Marginal Distributions")
    plot_histograms(scaled_df)

    # --- Correlation Analysis ---
    st.subheader("2. 🔗 Correlation Matrix (Spearman)")
    corr = plot_correlation_matrix(scaled_df)
    st.write("LHS generally reduces correlation between variables.")

    # --- Uniformity Metrics ---
    st.subheader("3. 📏 Uniformity Metrics")
    metrics = compare_with_random(scaled_df.values, n_trials=20)

    st.markdown(f"""
    **Discrepancy (lower is better):**
    - Your data: `{metrics['lhs_discrepancy']:.4f}`
    - Random: `{metrics['random_discrepancy_mean']:.4f}`

    **Maximin Distance (higher is better):**
    - Your data: `{metrics['lhs_maximin']:.4f}`
    - Random: `{metrics['random_maximin_mean']:.4f}`
    """)

    # --- Interpretation Help ---
    st.subheader("4. 🧠 Interpretation Guide")
    st.markdown("""
    - **Low correlation** between features suggests LHS-like stratification.
    - **Histograms** should appear uniform (flat) per feature.
    - **Pairwise plots** should avoid clustering and show space-filling patterns.
    - **Low discrepancy** and **high maximin** distance are indicative of LHS.
    - If your metrics outperform random sampling, the data may be LHS.
    """)

    # --- Final Determination ---
    st.subheader("5. 🏁 Final Determination")
    lhs_like, mean_corr = determine_sampling_type(metrics, corr)
    st.write(f"**Mean absolute correlation:** `{mean_corr:.4f}`")

    if lhs_like:
        st.success("✅ **The data appears to be LHS-like.**")
    else:
        if is_grid_sample(scaled_df.values):
            st.warning("⚠️ **The data appears to be generated from a structured grid.**")
        else:
            st.error("❌ **The data does not resemble LHS or a structured grid — likely random or unknown sampling.**")

else:
    st.info("⬆️ Upload a CSV file to begin.")
