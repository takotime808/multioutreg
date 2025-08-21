# Copyright (c) 2025 takotime808

import shap
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from sklearn.multioutput import MultiOutputRegressor
from multioutreg.utils.figure_utils import plot_to_b64
from typing import Any, List, Dict


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
        # Try a fast TreeExplainer; fall back to the model-agnostic Explainer
        try:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X)
        except Exception:
            explainer = shap.Explainer(estimator.predict, X)
            shap_values = explainer(X)
            # ``shap_values`` can be either an Explanation or ndarray
            shap_values = getattr(shap_values, "values", shap_values)

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


def generate_shap_plot(
    model: Any,
    X: np.ndarray,
    output_names: List[str]
) -> Dict[str, str]:
    """
    Generate SHAP summary plots for each output dimension.

    Parameters
    ----------
    model : Any
        Multi-output model with `estimators_` attribute.
    X : np.ndarray
        Input features used to compute SHAP values.
    output_names : List[str]
        Names of output dimensions.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping output names to base64-encoded SHAP plots.
    """
    plots = {}
    for i, name in enumerate(output_names):
        def plot_fn():
            est = model.estimators_[i]
            try:
                explainer = shap.Explainer(est.predict, X)  # safer, functional interface
                shap_values = explainer(X)
                shap.summary_plot(shap_values, X, show=False)
                plt.title(f"SHAP for {name}")
            except Exception as e:
                plt.figure()
                plt.text(0.5, 0.5, f"SHAP not supported for {type(est).__name__}", ha="center")
                plt.axis("off")
        plots[name] = plot_to_b64(plot_fn)
    return plots


def plot_multioutput_shap_bar_subplots_no_tree(
    model: MultiOutputRegressor,
    X: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    max_cols: int = 3,
    savefig: Optional[str] = None
) -> plt.Figure:
    import shap

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
        # KernelExplainer-compatible
        explainer = shap.Explainer(estimator.predict, X)
        shap_values = explainer(X)
        means = np.abs(shap_values.values).mean(axis=0)

        bars = axs[i].barh(feature_names, means)
        axs[i].set_title(output_names[i])
        axs[i].set_xlabel("Mean(|SHAP value|)")
        axs[i].invert_yaxis()
        for bar, value in zip(bars, means):
            axs[i].text(
                bar.get_width() + max(means) * 0.01,
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