# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import entropy
import umap


def infer_sampling_and_plot_umap(X: np.ndarray, explanation_indicator: bool = False):
    """Infer sampling method and return UMAP plot."""
    if X is None or len(X) == 0:
        raise ValueError("Input array X must be non-empty")

    X_scaled = StandardScaler().fit_transform(X)

    # --- UMAP embedding for visualization ---
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
    X_emb = reducer.fit_transform(X_scaled)

    # --- Compute heuristics on embedded space ---
    kd = KDTree(X_emb)
    dists = kd.query(X_emb, k=2)[0][:, 1]
    std = np.std(dists)
    clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(X_emb)
    sil = silhouette_score(X_emb, clusters)
    rng_min = dists.min()
    rng_max = dists.max()
    if rng_max - rng_min < 1e-6:
        ent = 0.0
    else:
        hist, _ = np.histogram(
            dists,
            bins=min(30, len(dists) // 2),
            range=(rng_min, rng_max),
            density=True,
        )
        ent = entropy(hist)

    if std < 0.05 and sil > 0.6:
        method = "Grid"
        explanation = "Low std and high silhouette -> Grid"
    elif std > 0.2 and sil < 0.3:
        method = "Random"
        explanation = "High std and low silhouette -> Random"
    elif 0.05 <= std <= 0.15:
        if ent < 2.0:
            method = "Sobol"
            explanation = "Moderate spread, low entropy -> Sobol"
        else:
            method = "LHS"
            explanation = "Moderate spread, higher entropy -> LHS"
    else:
        method = "Uncertain"
        explanation = "Pattern unclear"

    fig, ax = plt.subplots()
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=clusters, cmap="tab10", alpha=0.7)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"UMAP 2D Projection - Inferred: {method}")

    if not explanation_indicator:
        explanation = ""

    return method, fig, explanation