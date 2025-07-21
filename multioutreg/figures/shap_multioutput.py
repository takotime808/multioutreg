# Copyright (c) 2025 takotime808

import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from sklearn.multioutput import MultiOutputRegressor

def plot_multioutput_shap_bar_subplots(
    model: MultiOutputRegressor,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    max_cols: int = 3,
    savefig: Optional[str] = None
) -> plt.Figure:
    """
    Plot mean absolute SHAP value bar plots for each output of a MultiOutputRegressor in subplots.
    The figure always uses at most `max_cols` columns per row (default 3), and shows the SHAP value next to each bar.

    Parameters
    ----------
    model : MultiOutputRegressor
        A fitted MultiOutputRegressor model (e.g., wrapping RandomForestRegressor).
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features) used for SHAP analysis.
    feature_names : sequence of str, optional
        List of feature names (length n_features). If None, will use "feature_0", ... etc.
    output_names : sequence of str, optional
        List of output/target names (length n_outputs). If None, will use "Output 0", ... etc.
    max_cols : int, default=3
        Maximum number of columns per row in the subplot grid.
    savefig : str or None, default=None
        If given, path to save the figure to disk instead of showing.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure containing all subplots.

    Example
    -------
    >>> plot_multioutput_shap_bar_subplots(regr, X, feature_names=feature_names, output_names=output_names)
    """
    n_outputs = len(model.estimators_)
    n_features = X.shape[1]
    n_cols = min(max_cols, n_outputs)
    n_rows = int(np.ceil(n_outputs / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_outputs)]

    for i, estimator in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        means = np.abs(shap_values).mean(axis=0)
        bars = axs[i].barh(feature_names, means)
        axs[i].set_title(output_names[i])
        axs[i].set_xlabel("Mean(|SHAP value|)")
        axs[i].invert_yaxis()
        # Annotate: add value at right end of each bar
        for bar, value in zip(bars, means):
            axs[i].text(
                bar.get_width() + max(means) * 0.01,  # A small gap to the right
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va='center',
                ha='left',
                fontsize=10
            )

    for j in range(n_outputs, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    if savefig:
        fig.savefig(savefig)
        plt.close(fig)
    else:
        plt.show()
    return fig