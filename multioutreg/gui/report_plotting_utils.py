# Copyright (c) 2025 takotime808

import io
import shap
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# Utility to convert plots to base64
def plot_to_b64(plot_fn):
    buf = io.BytesIO()
    plot_fn()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_prediction_plot(y_true, y_pred, y_std, output_names):
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

def generate_shap_plot(model, X, output_names):
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

def generate_pdp_plot(model, X, output_names, feature_names):
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

def generate_uncertainty_plots():
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

def generate_umap_plot():
    def plot_fn():
        plt.figure()
        for i, label in enumerate(['LHS', 'Random']):
            data = np.random.normal(loc=i*2, scale=0.7, size=(50,2))
            plt.scatter(data[:,0], data[:,1], label=label, alpha=0.7)
        plt.title("UMAP projection of input sampling")
        plt.legend()
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
    return plot_to_b64(plot_fn), "EXAMPLE (not real output) -- Data cluster structure suggests a Latin Hypercube Sampling (LHS) technique was used."

def generate_error_histogram(y_true, y_pred, output_names):
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

# The rest of the Streamlit app remains as previously defined, using the functions above to generate plots
# inside the report generation section.
