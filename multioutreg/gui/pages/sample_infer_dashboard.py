# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")  # plotting helpers call plt.show(); never open desktop windows
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap.umap_ as umap

st.set_page_config(page_title="Sampling Bias Detection Dashboard", layout="wide")

def main():
    st.title("📊 Sampling Bias Detection Dashboard")
    st.markdown("Upload a **regression dataset** to analyze potential sampling bias and visualize data distributions.")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        # Column selectors
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(num_cols) < 2:
            st.warning("The dataset must have at least two numerical columns.")
            return

        selected_cols = st.multiselect("Select numeric columns for analysis", num_cols, default=num_cols[:5])

        # Analysis Tabs
        tabs = st.tabs([
            "📉 Sampling Pattern",
            "🔍 Pairwise KDE & Scatter",
            "🧮 Correlation Heatmap",
            "📈 Distributions",
            "❓ Missing Data",
            "📊 PCA Projection",
            "🧬 UMAP Projection"
        ])

        with tabs[0]:
            st.subheader("📉 Sampling Pattern Analysis")
            infer_sampling(df, selected_cols)

        with tabs[1]:
            st.subheader("🔍 Pairwise Relationships with KDE")
            plot_pairgrid_with_kde(df, selected_cols)

        with tabs[2]:
            st.subheader("🧮 Correlation Heatmap")
            show_correlation_heatmap(df, selected_cols)

        with tabs[3]:
            st.subheader("📈 Feature Distributions")
            show_feature_distributions(df, selected_cols)

        with tabs[4]:
            st.subheader("❓ Missing Data Overview")
            show_missing_data(df)

        with tabs[5]:
            st.subheader("📊 PCA Projection (2D)")
            plot_pca(df, selected_cols)

        with tabs[6]:
            st.subheader("🧬 UMAP Projection (2D)")
            plot_umap(df, selected_cols)

    else:
        st.info("Please upload a CSV file to begin.")

# === Visual Functions ===

def plot_pairgrid_with_kde(df, cols):
    if len(cols) < 2:
        st.warning("Select at least two numeric columns.")
        return
    g = sns.PairGrid(df[cols])
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot, cmap="Blues", fill=True)
    g.map_diag(sns.histplot, edgecolor="black")
    st.pyplot(g.fig, clear_figure=True)

def show_correlation_heatmap(df, cols):
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    st.pyplot(plt.gcf(), clear_figure=True)

def show_feature_distributions(df, cols):
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram - {col}")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot - {col}")
        st.pyplot(fig, clear_figure=True)

def infer_sampling(df, cols):
    unique_counts = [df[col].nunique() for col in cols]
    grid_like = np.prod(unique_counts) == len(df)
    lhs_like = all(df[col].value_counts().max() == 1 for col in cols)

    st.write("**Sampling Pattern Inference**")
    if grid_like:
        st.success("✔ The data resembles **Grid Sampling** (regular mesh with distinct fixed values per feature).")
    elif lhs_like:
        st.success("✔ The data resembles **Latin Hypercube Sampling (LHS)** (each value occurs only once, evenly spread).")
    else:
        st.warning("⚠ The sampling does not match Grid or LHS. Likely **random** or **custom sampling**.")

    st.write("**Unique value count per feature:**")
    st.dataframe(pd.DataFrame({"Feature": cols, "Unique Values": unique_counts}))

def show_missing_data(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("🎉 No missing values detected!")
    else:
        st.warning("⚠ Missing values found:")
        st.dataframe(missing.reset_index().rename(columns={"index": "Feature", 0: "Missing Values"}))

def plot_pca(df, cols):
    if len(cols) < 2:
        st.warning("PCA requires at least two numeric columns.")
        return
    X = df[cols].dropna()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(components[:, 0], components[:, 1], alpha=0.7)
    ax.set_title("PCA Projection (2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig, clear_figure=True)

def plot_umap(df, cols):
    if len(cols) < 2:
        st.warning("UMAP requires at least two numeric columns.")
        return
    X = df[cols].dropna()
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
    ax.set_title("UMAP Projection (2D)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    st.pyplot(fig, clear_figure=True)


# === Entry Point ===
if __name__ == "__main__":
    main()
