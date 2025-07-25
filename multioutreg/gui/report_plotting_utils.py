# Copyright (c) 2025 takotime808

import io
import shap
import umap
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Any
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans
from scipy.stats import entropy


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
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


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
        def plot_fn():
            plt.figure()
            plt.errorbar(y_true[:, i], y_pred[:, i], yerr=y_std[:, i], fmt='o', alpha=0.6)
            plt.plot(y_true[:, i], y_true[:, i], 'k--', label='Ideal')
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"{name}")
            plt.legend()
        plots[name] = plot_to_b64(plot_fn)
    return plots


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
                plt.text(0.5, 0.5, f"SHAP not supported for {type(est).__name__}", ha='center')
                plt.axis('off')
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

    Parameters
    ----------
    model : Any
        Multi-output model with `estimators_` attribute.
    X : np.ndarray
        Input features.
    output_names : List[str]
        Names of output dimensions.
    feature_names : List[str]
        Names of the input features.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping output names to base64-encoded PDP plots.
    """
    plots = {}
    for i, name in enumerate(output_names):
        def plot_fn():
            plt.figure()
            try:
                PartialDependenceDisplay.from_estimator(
                    model.estimators_[i], X, range(X.shape[1]), feature_names=feature_names, ax=plt.gca()
                )
                plt.title(f"PDP for {name}")
            except Exception as e:
                plt.text(0.5, 0.5, f"PDP not supported for {type(model.estimators_[i]).__name__}", ha='center')
                plt.axis('off')
        plots[name] = plot_to_b64(plot_fn)
    return plots


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
        plt.plot(probs, empirical, marker='o')
        plt.plot([0,1],[0,1],'k--', label='Ideal')
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
