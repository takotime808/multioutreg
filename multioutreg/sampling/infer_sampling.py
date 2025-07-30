# Copyright (c) 2025 takotime808

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from multioutreg.figures.umap_plot_classify import generate_umap_plot


def infer_sampling_and_plot_umap(
    X: np.ndarray,
    explanation_indicator: bool = False,
) -> Tuple[str, plt.Figure] | Tuple[str, plt.Figure, str]:
    """
    Wrapper around generate_umap_plot to return method, plot, and explanation.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix.
    explanation_indicator : bool
        If True, also return the explanation text.

    Returns
    -------
    Union[Tuple[str, plt.Figure], Tuple[str, plt.Figure, str]]
        Sampling method name and matplotlib Figure (and explanation if requested).
    """
    b64img, explanation = generate_umap_plot(X)
    method = explanation.split("->")[-1].strip() if "->" in explanation else "Uncertain"

    # generate figure manually again for consistent UI (only if explanation_indicator)
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=0)
    X_emb = reducer.fit_transform(StandardScaler().fit_transform(X))
    clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(X_emb)

    fig, ax = plt.subplots()
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=clusters, cmap="tab10", alpha=0.7)
    ax.set_title(f"UMAP 2D Projection (Inferred: {method})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True)

    if explanation_indicator:
        return method, fig, explanation
    else:
        return method, fig
