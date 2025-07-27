# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import base64
import io
from jinja2 import Environment, FileSystemLoader

from multioutreg.utils.figure_utils import (
    safe_plot_b64,
    # plot_to_b64,
)

# # # ---- Helper to convert plot to base64 ----
# def plot_to_b64():
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     img_b64 = base64.b64encode(buf.read()).decode('utf-8')
#     return img_b64

# ---- Fake Data ----
np.random.seed(42)
n_samples = 100
y_true = np.random.normal(size=(n_samples, 2))
y_pred = y_true + np.random.normal(scale=0.1, size=y_true.shape)
y_std = np.abs(np.random.normal(scale=0.15, size=y_true.shape))
output_names = ["Pressure", "Temperature"]

# ---- Metrics ----
metrics = {
    "Pressure": {"r2": 0.91, "rmse": 0.12, "mae": 0.09, "max_error": 0.3},
    "Temperature": {"r2": 0.88, "rmse": 0.18, "mae": 0.13, "max_error": 0.4},
}

uncertainty_metrics = {"NLL": 1.12, "Sharpness": 0.19, "Miscoverage": 0.06, "Calibration Error": 0.04}

# ---- Plots ----
# Predictions vs True (with error bars)
prediction_plots = {}
for i, name in enumerate(output_names):
    def prediction_plot():
        plt.figure()
        plt.errorbar(y_true[:,i], y_pred[:,i], yerr=y_std[:,i], fmt='o', alpha=0.6)
        plt.plot(y_true[:,i], y_true[:,i], 'k--', label='Ideal')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{name}")
        plt.legend()
    prediction_plots[name] = safe_plot_b64(prediction_plot)

# Fake SHAP plots
shap_plots = {}
X = np.random.uniform(-1,1,(n_samples,3))
for i, name in enumerate(output_names):
    def shap_plot():
        explainer = shap.Explainer(lambda X: X[:,0] + 0.5*X[:,1] + 0.2*X[:,2], X)
        shap_values = explainer(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP for {name}")
    shap_plots[name] = safe_plot_b64(shap_plot)

# Fake PDP plots
pdp_plots = {}
for i, name in enumerate(output_names):
    def pdp_plot():
        plt.figure()
        x = np.linspace(-1, 1, 50)
        plt.plot(x, x + (i+1)*0.2, label="Feature 1")
        plt.plot(x, -x + (i+1)*0.1, label="Feature 2")
        plt.xlabel("Feature Value")
        plt.ylabel("Partial Dependence")
        plt.title(f"PDP for {name}")
        plt.legend()
    pdp_plots[name] = safe_plot_b64(pdp_plot)

# Uncertainty toolbox style plots (calibration curve example)
uncertainty_plots = []
def plot_uncert():
    probs = np.linspace(0,1,11)
    empirical = probs + np.random.normal(scale=0.03, size=probs.shape)
    plt.plot(probs, empirical, marker='o')
    plt.plot([0,1],[0,1],'k--', label='Ideal')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Empirical Probability")
    plt.title("Calibration Curve")
    plt.legend()
uncertainty_plots.append({
    "img_b64": safe_plot_b64(plot_uncert),
    "title": "Calibration Curve",
    "caption": "Shows calibration of predicted uncertainty."
})

# Sampling UMAP plot (just a scatter)
def umap_plot():
    plt.figure()
    for i, label in enumerate(['LHS', 'Random']):
        data = np.random.normal(loc=i*2, scale=0.7, size=(50,2))
        plt.scatter(data[:,0], data[:,1], label=label, alpha=0.7)
    plt.title("UMAP projection of input sampling")
    plt.legend()
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
sampling_umap_plot = safe_plot_b64(umap_plot)
sampling_method_explanation = "Data cluster structure suggests a Latin Hypercube Sampling (LHS) technique was used."

# Other diagnostic plot example
other_plots = []
def other_plot():
    plt.figure()
    plt.hist(y_pred[:,0] - y_true[:,0], bins=20, alpha=0.8)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Histogram (Pressure)")

other_plots.append({
    "img_b64": safe_plot_b64(other_plot),
    "title": "Error Histogram", "caption":
    "Histogram of prediction errors for Pressure."
})

# ---- Prepare context for template ----
context = dict(
    project_title="Multi-Fidelity, Multi-Output Surrogate Model Demo Report",
    model_type="Multi-Fidelity Gaussian Process",
    fidelity_levels=['Low', 'Medium', 'High'],
    output_names=output_names,
    description="Demonstration of a surrogate model with multiple fidelities and outputs, including uncertainty quantification.",
    metrics=metrics,
    uncertainty_metrics=uncertainty_metrics,
    uncertainty_plots=uncertainty_plots,
    prediction_plots=prediction_plots,
    shap_plots=shap_plots,
    pdp_plots=pdp_plots,
    sampling_umap_plot=sampling_umap_plot,
    sampling_method_explanation=sampling_method_explanation,
    sampling_other_plots=[],
    other_plots=other_plots,
    n_train=70,
    n_test=30,
    cross_validation="5-fold",
    seed=42,
    notes="All results are for demonstration purposes only.\nNo real-world data used."
)

output_path = "outputs/surrogate_report_example_safe_fig.html"
template_path = "multioutreg/report/template.html"

# ---- Jinja2 rendering ----
env = Environment(loader=FileSystemLoader("."))
template = env.get_template(template_path)
html = template.render(**context)

with open(output_path, "w") as f:
    f.write(html)

print(f"Report written to {output_path}")
