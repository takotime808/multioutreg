# Copyright (c) 2025 takotime808

import umap
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from multioutreg.gui.report_plotting_utils import (
    plot_to_b64,
)
from multioutreg.sampling.detect import detect_sampling_method
from multioutreg.sampling.pca_sampling_detector import label_colour_values, pca_embedding


def generate_umap_plot(
    X,
    random_state: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    labels=None,
):
    """
    Plot a UMAP projection of the inputs and infer the sampling method.

    The inferred method comes from :func:`multioutreg.sampling.detect.detect_sampling_method`
    (Monte Carlo tests of stratification, discrepancy and nearest-neighbour
    spacing against simulated designs). UMAP is only used for the picture: it
    does not preserve distances, so its embedding cannot tell designs apart.
    If UMAP fails, the plot falls back to a PCA projection.

    Parameters
    ----------
    X : np.ndarray
        Input (design) matrix. Pass only input columns, not responses.
    random_state : int
        Seed for UMAP and for the simulated reference designs.
    n_neighbors : int
        UMAP neighbourhood size (capped at ``n - 1``).
    min_dist : float
        UMAP minimum distance between embedded points.
    labels : array-like or None
        Optional per-point values to colour the points by.

    Returns
    -------
    Tuple[str, str]
        Base64-encoded plot and inferred sampling explanation (ends in ``-> <method>``).
    """
    if X is None or len(X) == 0:
        raise ValueError("generate_umap_plot needs a non-empty input matrix")
    X = np.asarray(X, dtype=float)

    try:
        detection = detect_sampling_method(X, random_state=random_state)
        method, explanation = detection.method, detection.explanation
    except ValueError as exc:  # e.g. fewer than 4 points
        method, explanation = "Uncertain", f"Sampling method not tested: {exc} -> Uncertain"

    X_std = StandardScaler().fit_transform(X)
    projection = "UMAP"
    if X.shape[1] >= 2 and X.shape[0] > 3:
        try:
            reducer = umap.UMAP(
                n_neighbors=min(n_neighbors, X.shape[0] - 1),
                min_dist=min_dist,
                n_components=2,
                random_state=random_state,
            )
            X_emb = reducer.fit_transform(X_std)
            xlabel_default, ylabel_default = "UMAP-1", "UMAP-2"
        except Exception as exc:
            warnings.warn(f"UMAP failed ({exc}); plotting a PCA projection instead.")
            projection = "PCA"
            X_emb, _ = pca_embedding(X, random_state=random_state)
            xlabel_default, ylabel_default = "PC1", "PC2"
    else:
        X_emb = np.column_stack([X_std[:, 0], np.zeros(len(X_std))])
        xlabel_default, ylabel_default = "x (standardised)", ""

    def plot_fn(xlabel: str = xlabel_default, ylabel: str = ylabel_default):
        plt.figure()
        if labels is None:
            plt.scatter(X_emb[:, 0], X_emb[:, 1], alpha=0.7)
        else:
            plt.scatter(X_emb[:, 0], X_emb[:, 1], c=label_colour_values(labels), cmap="viridis", alpha=0.7)
            plt.colorbar()
        plt.title(f"{projection} 2D Projection - Inferred: {method}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return plot_to_b64(plot_fn), explanation
