# Copyright (c) 2025 takotime808

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from multioutreg.sampling.detect import detect_sampling_method


def infer_sampling_and_plot_umap(
    X: np.ndarray,
    explanation_indicator: bool = False,
    random_state: int = 0,
) -> Tuple[str, plt.Figure] | Tuple[str, plt.Figure, str]:
    """
    Infer the sampling method of ``X`` and plot its UMAP projection.

    The method comes from :func:`multioutreg.sampling.detect.detect_sampling_method`;
    the UMAP projection is for display only.

    Parameters
    ----------
    X : np.ndarray
        Input (design) matrix. Pass only input columns, not responses.
    explanation_indicator : bool
        If True, also return the explanation text.
    random_state : int
        Seed for UMAP and for the simulated reference designs.

    Returns
    -------
    Union[Tuple[str, plt.Figure], Tuple[str, plt.Figure, str]]
        Sampling method name and matplotlib Figure (and explanation if requested).
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    detection = detect_sampling_method(X, random_state=random_state)
    method, explanation = detection.method, detection.explanation

    n_neighbors = min(15, X.shape[0] - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=random_state)
    X_emb = reducer.fit_transform(StandardScaler().fit_transform(X))

    fig, ax = plt.subplots()
    ax.scatter(X_emb[:, 0], X_emb[:, 1], alpha=0.7)
    ax.set_title(f"UMAP 2D Projection (Inferred: {method})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)

    if explanation_indicator:
        return method, fig, explanation
    else:
        return method, fig
