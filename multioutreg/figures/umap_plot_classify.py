# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import qmc, entropy
import umap
import warnings
import os

from multioutreg.gui.report_plotting_utils import plot_to_b64


def classify_sampling_method(std, sil, ent, n_samples, method_used="UMAP"):
    if method_used == "PCA":
        if std < 0.01 and sil > 0.3 and ent < 2.2:
            return "Grid", "PCA: low std, mild structure"
        elif std > 0.04 and sil < 0.4 and ent > 2.9:
            return "Random", "PCA: wide spread, high entropy → random"
        elif 0.035 <= std <= 0.07 and ent < 2.6:
            return "Sobol", "PCA: moderate spread, low entropy"
        elif 0.035 <= std <= 0.07 and ent >= 2.6:
            return "LHS", "PCA: moderate spread, high entropy"
        elif std == 0.0:
            return "Uncertain", "PCA: flat structure"
        else:
            return "Uncertain", "PCA: ambiguous pattern"
    else:
        if std < 0.06 and sil > 0.5 and ent < 2.5:
            return "Grid", "UMAP: tightly packed + high cluster structure"
        elif std > 0.25 and sil < 0.25 and ent > 2.5:
            return "Random", "UMAP: high spread, low structure"
        elif 0.09 <= std <= 0.16 and ent < 2.0:
            return "Sobol", "UMAP: moderate spread, low entropy"
        elif 0.09 <= std <= 0.17 and ent >= 2.0:
            return "LHS", "UMAP: moderate spread, high entropy"
        else:
            return "Uncertain", "UMAP: ambiguous pattern"


def generate_auto_projection_classification_metrics(X, use_umap=True, random_state=42):
    X = np.asarray(X)
    n_samples = X.shape[0]
    X_scaled = StandardScaler().fit_transform(X)

    method_used = "UMAP"
    try:
        if use_umap:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
            X_emb = reducer.fit_transform(X_scaled)
        else:
            raise ImportError  # force fallback
    except Exception:
        method_used = "PCA"
        reducer = PCA(n_components=2, random_state=random_state)
        X_emb = reducer.fit_transform(X_scaled)

    clusters = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X_emb)
    dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
    std = np.std(dists)
    sil = silhouette_score(X_emb, clusters) if len(np.unique(clusters)) > 1 else 0.0
    ent = entropy(np.histogram(dists, bins=30, density=True)[0])

    method, explanation = classify_sampling_method(std, sil, ent, n_samples, method_used)
    return {
        "method": method,
        "explanation": explanation,
        "projection": method_used,
        "std": std,
        "silhouette": sil,
        "entropy": ent
    }


def run_auto_sampling_tests():
    test_cases = {
        "Grid": np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), -1).reshape(-1, 2),
        "Random": np.random.rand(400, 2),
        "Sobol": qmc.Sobol(d=2, scramble=False).random(256),
        "LHS": qmc.LatinHypercube(d=2).random(400),
        "Uncertain": np.ones((50, 2))
    }

    results = {}
    for label, X in test_cases.items():
        result = generate_auto_projection_classification_metrics(X, use_umap=True)
        results[label] = result

    return results


def generate_umap_plot(
    X: np.ndarray | pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    labels=None,
    random_state: int = 0,
    savefig: str | None = None,
):
    """
    Generate a 2D projection plot (UMAP or PCA fallback) and classify the sampling method.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
    n_neighbors : int
        UMAP n_neighbors.
    min_dist : float
        UMAP min_dist.
    labels : array-like or None
        Labels to color the points by.
    random_state : int
        Random seed.
    save_path : str or None
        If provided, saves the image as PNG to this path.

    Returns
    -------
    Tuple[str, dict]
        - base64 PNG image string
        - classification dictionary
    """
    if X is None or len(X) == 0:
        X = np.random.normal(size=(100, 5))

    X = np.asarray(X)
    n_samples = X.shape[0]
    X_scaled = StandardScaler().fit_transform(X)

    # Try UMAP, fall back to PCA
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
        X_emb = reducer.fit_transform(X_scaled)
        method_used = "UMAP"
    except Exception as e:
        warnings.warn(f"UMAP failed ({e}), falling back to PCA.")
        reducer = PCA(n_components=2, random_state=random_state)
        X_emb = reducer.fit_transform(X_scaled)
        method_used = "PCA"

    clusters = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X_emb)
    dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
    std = np.std(dists)
    sil = silhouette_score(X_emb, clusters) if len(np.unique(clusters)) > 1 else 0.0
    ent = entropy(np.histogram(dists, bins=30, density=True)[0])

    method, explanation = classify_sampling_method(std, sil, ent, n_samples, method_used)

    def plot_fn(xlabel="Dim 1", ylabel="Dim 2"):
        plt.figure(figsize=(7, 6))
        c = labels if labels is not None else clusters
        scatter = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=c, cmap="tab10", alpha=0.7, edgecolor="k", linewidth=0.4)

        if labels is not None:
            unique_labels = np.unique(labels)
            handles, _ = scatter.legend_elements()
            plt.legend(handles, unique_labels, title="Labels", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.title(f"{method_used} Projection — Inferred: {method}", fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        if savefig:
            os.makedirs(os.path.dirname(savefig), exist_ok=True)
            plt.savefig(savefig, dpi=300, bbox_inches="tight")

    img_b64 = plot_to_b64(plot_fn)

    return img_b64, {
        "method": method,
        "explanation": explanation,
        "projection": method_used,
        "std": float(std),
        "silhouette": float(sil),
        "entropy": float(ent)
    }