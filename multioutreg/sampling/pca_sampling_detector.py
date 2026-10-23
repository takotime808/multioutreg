# Copyright (c) 2025 takotime808

"""PCA view of a design of experiments, labelled with its inferred sampling method.

PCA is linear and deterministic, so unlike UMAP it keeps relative distances and
needs no extra dependency, which makes it a reasonable default picture of a
design. The method label still comes from
:func:`multioutreg.sampling.detect.detect_sampling_method`: thresholds on the
2-D embedding (nearest-neighbour spread, silhouette, entropy) depend on ``n``
and ``d`` and did not separate the designs reliably, so those numbers are
reported as diagnostics only.
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

from multioutreg.sampling.detect import detect_sampling_method


def pca_embedding(X: np.ndarray, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Standardise ``X`` and project it onto (at most) two principal components.

    Returns ``(X_emb, explained_variance_ratio)``. ``X_emb`` always has two
    columns; a single-input design is padded with zeros. Constant data embeds
    at the origin with zero explained variance.
    """
    X = np.asarray(X, dtype=float)
    X_std = StandardScaler().fit_transform(X)
    n_comp = min(2, X.shape[1], X.shape[0])
    if not np.any(X_std):
        return np.zeros((X.shape[0], 2)), np.zeros(n_comp)
    pca = PCA(n_components=n_comp, random_state=random_state).fit(X_std)
    X_emb = pca.transform(X_std)
    if n_comp < 2:
        X_emb = np.column_stack([X_emb, np.zeros((X.shape[0], 2 - n_comp))])
    return X_emb, pca.explained_variance_ratio_


def embedding_diagnostics(X_emb: np.ndarray, random_state: int = 0) -> Dict[str, float]:
    """Structure of a 2-D embedding: nearest-neighbour spread, 2-means silhouette, entropy.

    Descriptive only; see the module docstring for why they are not used to
    classify the design.
    """
    n_distinct = np.unique(X_emb, axis=0).shape[0]
    dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
    sil = 0.0
    if n_distinct >= 3:
        clusters = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(X_emb)
        if len(np.unique(clusters)) > 1:
            sil = float(silhouette_score(X_emb, clusters))
    return {
        "std": float(np.std(dists)),
        "silhouette": sil,
        "entropy": float(entropy(np.histogram(dists, bins=30, density=True)[0])),
    }


def generate_sampling_plot_and_metrics(
    X: np.ndarray,
    random_state: int = 42,
    labels: Optional[np.ndarray] = None,
) -> Tuple[plt.Figure, dict]:
    """Plot the PCA projection of a design and infer its sampling method.

    Parameters
    ----------
    X : np.ndarray
        Input (design) matrix. Pass only input columns, not responses.
    random_state : int
        Seed for PCA, the diagnostics and the simulated reference designs.
    labels : array-like or None
        Optional per-point values to colour the points by.

    Returns
    -------
    Tuple[plt.Figure, dict]
        The figure and a dict with ``method``, ``explanation``, ``projection``
        (``"PCA"``), ``explained_variance``, ``p_values`` and the embedding
        diagnostics ``std``, ``silhouette`` and ``entropy``.
    """
    if X is None or len(X) == 0:
        raise ValueError("generate_sampling_plot_and_metrics needs a non-empty input matrix")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    try:
        detection = detect_sampling_method(X, random_state=random_state)
        method, explanation, p_values = detection.method, detection.explanation, detection.p_values
    except ValueError as exc:  # e.g. fewer than 4 points
        method, explanation, p_values = "Uncertain", f"Sampling method not tested: {exc} -> Uncertain", {}

    X_emb, evr = pca_embedding(X, random_state=random_state)

    fig, ax = plt.subplots(figsize=(6, 5))
    if labels is None:
        ax.scatter(X_emb[:, 0], X_emb[:, 1], alpha=0.7, edgecolor="k", linewidth=0.3)
    else:
        sc = ax.scatter(X_emb[:, 0], X_emb[:, 1], c=label_colour_values(labels), cmap="viridis",
                        alpha=0.7, edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax)
    ax.set_title(f"PCA Projection - Inferred: {method}")
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)" if len(evr) > 0 else "PC1")
    ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)" if len(evr) > 1 else "")
    fig.tight_layout()

    return fig, {
        "method": method,
        "explanation": explanation,
        "projection": "PCA",
        "explained_variance": [float(v) for v in evr],
        "p_values": {k: float(v) for k, v in p_values.items()},
        **embedding_diagnostics(X_emb, random_state=random_state),
    }


def label_colour_values(labels) -> np.ndarray:
    """Numeric colour values for ``labels`` (categories become their codes)."""
    labels = np.asarray(labels)
    if np.issubdtype(labels.dtype, np.number):
        return labels.astype(float)
    return np.unique(labels.astype(str), return_inverse=True)[1]
