# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from multioutreg.utils.figure_utils import plot_to_b64

def generate_pca_variance_plot(
    pca: PCA,
    n_selected: int | None = None,
    threshold: float | None = None,
) -> str:
    """
    Generate a base64-encoded bar plot showing the explained variance ratio of each principal component.

    Parameters
    ----------
    pca : PCA
        Fitted PCA instance.
    n_selected : int | None, optional
        Number of components retained. If provided, a vertical dashed line is
        drawn on the scree plot.
    threshold : float | None, optional
        Explained variance threshold. When given, a horizontal dashed line is
        drawn to illustrate the cutoff.

    Returns
    -------
    str
       Base64 encoded PNG image of the scree plot.
    """
    def plot_fn():
        comps = np.arange(1, len(pca.explained_variance_ratio_) + 1)
        height = max(4, 0.3 * len(comps))  # Dynamic height
        plt.figure(figsize=(int(1.5*len(comps)), height))
        plt.plot(comps, pca.explained_variance_ratio_, marker="o")
        plt.xlabel("Principal Component", fontsize=10)
        plt.ylabel("Explained Variance Ratio", fontsize=10)
        plt.yticks(comps, [f"PC{i}" for i in comps])  # Smaller y-axis font
        plt.title("PCA Scree Plot")
        if n_selected is not None:
            plt.axvline(n_selected, color="red", linestyle="--", label=f"k = {n_selected}")
        if threshold is not None:
            plt.axhline(threshold, color="green", linestyle=":", label=f"threshold = {threshold:.2f}")
        if n_selected is not None or threshold is not None:
            plt.legend()
        plt.tight_layout()

    return plot_to_b64(plot_fn)
