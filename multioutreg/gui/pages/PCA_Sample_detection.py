# Copyright (c) 2025 takotime808

import numpy as np
import streamlit as st
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KDTree
# from sklearn.metrics import silhouette_score
# from scipy.stats import entropy
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

from multioutreg.sampling.pca_sampling_detector import (
    # classify_sampling_method,
    generate_sampling_plot_and_metrics,
)

# def classify_sampling_method(std, sil, ent, n_samples, method_used="PCA"):
#     if method_used == "PCA":
#         if std < 0.01 and sil > 0.3 and ent < 2.2:
#             return "Grid", "PCA: low std, mild structure"
#         elif std > 0.04 and sil < 0.4 and ent > 2.9:
#             return "Random", "PCA: wide spread, high entropy → random"
#         elif 0.035 <= std <= 0.07 and ent < 2.8:
#             return "Sobol", "PCA: moderate spread, low entropy"
#         elif 0.035 <= std <= 0.07 and ent >= 2.8:
#             return "LHS", "PCA: moderate spread, high entropy"
#         elif std == 0.0:
#             return "Uncertain", "PCA: flat structure"
#         else:
#             return "Uncertain", "PCA: ambiguous pattern"
#     else:
#         return "Uncertain", "Fallback projection not supported"


# def generate_sampling_plot_and_metrics(X, random_state=42):
#     X = np.asarray(X)
#     n_samples = X.shape[0]
#     X_scaled = StandardScaler().fit_transform(X)

#     reducer = PCA(n_components=2, random_state=random_state)
#     X_emb = reducer.fit_transform(X_scaled)

#     clusters = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X_emb)
#     dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
#     std = np.std(dists)
#     sil = silhouette_score(X_emb, clusters) if len(np.unique(clusters)) > 1 else 0.0
#     ent = entropy(np.histogram(dists, bins=30, density=True)[0])

#     method, explanation = classify_sampling_method(std, sil, ent, n_samples, method_used="PCA")

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.scatter(X_emb[:, 0], X_emb[:, 1], c=clusters, cmap="tab10", alpha=0.7, edgecolor="k", linewidth=0.3)
#     ax.set_title(f"PCA Projection — Inferred: {method}")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     plt.tight_layout()

#     return fig, {
#         "method": method,
#         "explanation": explanation,
#         "projection": "PCA",
#         "std": float(std),
#         "silhouette": float(sil),
#         "entropy": float(ent)
#     }


# Streamlit app tab integration
st.title("Sampling Method Detection (PCA-based)")

with st.sidebar:
    random_state = st.number_input("Random State", min_value=0, value=42, step=1)

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        st.warning("No numeric data found in file.")
    else:
        fig, metrics = generate_sampling_plot_and_metrics(numeric_df.values, random_state=random_state)

        tab1, tab2 = st.tabs(["📊 PCA Plot", "📈 Metrics"])
        with tab1:
            st.pyplot(fig)
        with tab2:
            st.json(metrics)
else:
    st.info("Please upload a CSV file to begin analysis.")
