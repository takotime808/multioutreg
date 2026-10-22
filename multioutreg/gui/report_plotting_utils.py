# Copyright (c) 2025 takotime808

import io
# import shap
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Any
# from sklearn.decomposition import PCA


# Utility to convert plots to base64
def plot_to_b64(plot_fn: Callable[[], None]) -> str:
    """
    Convert a plot function to a base64-encoded PNG image.

    Parameters
    ----------
    plot_fn : Callable[[], None]
        Function that generates a matplotlib plot.

    Returns
    -------
    str
        Base64-encoded string of the PNG plot image.
    """
    buf = io.BytesIO()
    plot_fn()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_prediction_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    output_names: List[str]
) -> Dict[str, str]:
    """
    Generate base64-encoded scatter plots of predictions with error bars.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples, n_outputs).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples, n_outputs).
    y_std : np.ndarray
        Standard deviations of predictions of shape (n_samples, n_outputs).
    output_names : List[str]
        Names of output dimensions.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping output names to base64-encoded plot images.
    """
    plots = {}
    for i, name in enumerate(output_names):

        def plot_fn(alpha=0.6):
            plt.figure()
            plt.errorbar(y_true[:, i], y_pred[:, i], yerr=y_std[:, i], fmt="o", alpha=alpha)
            plt.plot(y_true[:, i], y_true[:, i], "k--", label="Ideal")
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"{name}")
            plt.legend()

        plots[name] = plot_to_b64(plot_fn)
    return plots


def generate_pdp_plot(
    model: Any,
    X: np.ndarray,
    output_names: List[str],
    feature_names: List[str]
) -> Dict[str, str]:
    """
    Generate partial dependence plots (PDPs) for each output dimension.

    Kept for backwards compatibility; see
    :func:`multioutreg.figures.pdp_plots.generate_pdp_plot`.
    """
    # Imported here: pdp_plots imports plot_to_b64 from this module.
    from multioutreg.figures.pdp_plots import generate_pdp_plot as _generate_pdp_plot
    return _generate_pdp_plot(model, X, output_names, feature_names)


def generate_uncertainty_plots() -> List[Dict[str, str]]:
    """
    Generate a dummy calibration curve for uncertainty illustration.

    Returns
    -------
    List[Dict[str, str]]
        List containing a single dictionary with base64-encoded plot,
        title, and caption.
    """
    plots = []
    def plot_fn():
        probs = np.linspace(0, 1, 11)
        empirical = probs + np.random.normal(scale=0.03, size=probs.shape)
        plt.plot(probs, empirical, marker="o")
        plt.plot([0,1],[0,1],"k--", label="Ideal")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Empirical Probability")
        plt.title("Calibration Curve")
        plt.legend()
    plots.append({
        "img_b64": plot_to_b64(plot_fn),
        "title": "Calibration Curve",
        "caption": "Shows calibration of predicted uncertainty."
    })
    return plots


def generate_umap_plot(X):
    """
    Generate a UMAP projection and infer the sampling method.

    Kept for backwards compatibility; see
    :func:`multioutreg.figures.umap_plot_classify.generate_umap_plot`.
    """
    # Imported here: umap_plot_classify imports plot_to_b64 from this module.
    from multioutreg.figures.umap_plot_classify import generate_umap_plot as _generate_umap_plot
    return _generate_umap_plot(X)


def generate_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_names: List[str]
) -> List[Dict[str, str]]:
    """
    Generate error histograms for each output dimension.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    output_names : List[str]
        Names of output dimensions.

    Returns
    -------
    List[Dict[str, str]]
        List of dictionaries containing base64-encoded histograms and metadata.
    """
    plots = []
    for i, name in enumerate(output_names):
        def plot_fn():
            plt.figure()
            plt.hist(y_pred[:, i] - y_true[:, i], bins=20, alpha=0.8)
            plt.xlabel("Prediction Error")
            plt.ylabel("Frequency")
            plt.title(f"Error Histogram ({name})")
        plots.append({
            "img_b64": plot_to_b64(plot_fn),
            "title": f"Error Histogram", 
            "caption": f"Histogram of prediction errors for {name}."
        })
    return plots
