# Copyright (c) 2025 takotime808

import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.stats import entropy

from multioutreg.gui.report_plotting_utils import plot_to_b64


def generate_umap_plot(X):
    """
    Generate a UMAP projection and infer the sampling method visually.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.

    Returns
    -------
    Tuple[str, str]
        Base64-encoded plot and inferred sampling explanation.
    """

    if X is None or len(X) == 0:
        X = np.random.normal(size=(100, 5))

    # UMAP embedding
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
    X_scaled = StandardScaler().fit_transform(X)
    X_emb = reducer.fit_transform(X_scaled)

    # Distance & clustering metrics
    dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
    std_dist = np.std(dists)
    hist_vals, _ = np.histogram(dists, bins=30, density=True)
    dist_entropy = entropy(hist_vals + 1e-8)

    clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(X_emb)
    sil_score = silhouette_score(X_emb, clusters)

    # Sampling method classification logic
    if std_dist < 0.05 and sil_score > 0.6:
        method = "Grid"
        explanation = f"Grid-like sampling detected — very low spread (std={std_dist:.3f}), high cluster separation (silhouette={sil_score:.2f})"
    elif std_dist > 0.2 and sil_score < 0.3:
        method = "Random"
        explanation = f"Random-like sampling — high spread (std={std_dist:.3f}), weak clustering (silhouette={sil_score:.2f})"
    elif 0.05 <= std_dist <= 0.15:
        if dist_entropy < 2.0:
            method = "Sobol"
            explanation = f"Low-entropy structure (entropy={dist_entropy:.2f}) suggests Sobol sequence sampling"
        else:
            method = "LHS"
            explanation = f"Moderate spread and high entropy (entropy={dist_entropy:.2f}) suggest Latin Hypercube Sampling (LHS)"
    else:
        method = "Uncertain"
        explanation = f"Pattern unclear — std={std_dist:.3f}, entropy={dist_entropy:.2f}, silhouette={sil_score:.2f}"

    # Plotting function
    def plot_fn(xlabel: str = "UMAP-1", ylabel: str = "UMAP-2"):
        plt.figure()
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=clusters, cmap="tab10", alpha=0.7)
        plt.title(f"UMAP 2D Projection - Inferred: {method}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return plot_to_b64(plot_fn), explanation
