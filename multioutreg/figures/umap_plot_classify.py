# Copyright (c) 2025 takotime808

import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.stats import entropy

from multioutreg.gui.report_plotting_utils import (
    plot_to_b64,
)


def generate_umap_plot(X):
    """
    Generate a UMAP projection and infer the sampling method.

    Parameters
    ----------
    X : np.ndarray or None
        Input feature matrix.

    Returns
    -------
    Tuple[str, str]
        Base64-encoded plot and inferred sampling explanation.
    """

    if X is None or len(X) == 0:
        X = np.random.normal(size=(100, 5))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
    X_emb = reducer.fit_transform(StandardScaler().fit_transform(X))

    clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(X_emb)

    dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
    std = np.std(dists)
    sil = silhouette_score(X_emb, clusters)
    ent = entropy(np.histogram(dists, bins=30, density=True)[0])

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

    def plot_fn(xlabel: str = "UMAP-1", ylabel: str = "UMAP-2"):
        plt.figure()
        plt.scatter(X_emb[:, 0], X_emb[:, 1], c=clusters, cmap="tab10", alpha=0.7)
        plt.title(f"UMAP 2D Projection - Inferred: {method}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return plot_to_b64(plot_fn), explanation


