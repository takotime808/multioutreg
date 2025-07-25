# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from jinja2 import Template

from multioutreg.gui.report_plotting_utils import (
    # plot_to_b64,
    generate_prediction_plot,
    generate_shap_plot,
    generate_pdp_plot,
    generate_uncertainty_plots,
    generate_umap_plot,
    generate_error_histogram
)

# # Utility to convert plots to base64
# def plot_to_b64(plot_fn):
#     buf = io.BytesIO()
#     plot_fn()
#     plt.savefig(buf, format='png', bbox_inches='tight')
#     plt.close()
#     buf.seek(0)
#     return base64.b64encode(buf.read()).decode('utf-8')

# def generate_prediction_plot(y_true, y_pred, y_std, output_names):
#     plots = {}
#     for i, name in enumerate(output_names):
#         def plot_fn():
#             plt.figure()
#             plt.errorbar(y_true[:, i], y_pred[:, i], yerr=y_std[:, i], fmt='o', alpha=0.6)
#             plt.plot(y_true[:, i], y_true[:, i], 'k--', label='Ideal')
#             plt.xlabel("True")
#             plt.ylabel("Predicted")
#             plt.title(f"{name}")
#             plt.legend()
#         plots[name] = plot_to_b64(plot_fn)
#     return plots

# def generate_shap_plot(model, X, output_names):
#     plots = {}
#     for i, name in enumerate(output_names):
#         def plot_fn():
#             est = model.estimators_[i]
#             try:
#                 explainer = shap.Explainer(est.predict, X)  # safer, functional interface
#                 shap_values = explainer(X)
#                 shap.summary_plot(shap_values, X, show=False)
#                 plt.title(f"SHAP for {name}")
#             except Exception as e:
#                 plt.figure()
#                 plt.text(0.5, 0.5, f"SHAP not supported for {type(est).__name__}", ha='center')
#                 plt.axis('off')
#         plots[name] = plot_to_b64(plot_fn)
#     return plots


# def generate_pdp_plot(output_names):
#     plots = {}
#     for i, name in enumerate(output_names):
#         def plot_fn():
#             plt.figure()
#             x = np.linspace(-1, 1, 50)
#             plt.plot(x, x + (i+1)*0.2, label="Feature 1")
#             plt.plot(x, -x + (i+1)*0.1, label="Feature 2")
#             plt.xlabel("Feature Value")
#             plt.ylabel("Partial Dependence")
#             plt.title(f"PDP for {name}")
#             plt.legend()
#         plots[name] = plot_to_b64(plot_fn)
#     return plots

# def generate_uncertainty_plots():
#     plots = []
#     def plot_fn():
#         probs = np.linspace(0, 1, 11)
#         empirical = probs + np.random.normal(scale=0.03, size=probs.shape)
#         plt.plot(probs, empirical, marker='o')
#         plt.plot([0,1],[0,1],'k--', label='Ideal')
#         plt.xlabel("Predicted Probability")
#         plt.ylabel("Empirical Probability")
#         plt.title("Calibration Curve")
#         plt.legend()
#     plots.append({
#         "img_b64": plot_to_b64(plot_fn),
#         "title": "Calibration Curve",
#         "caption": "Shows calibration of predicted uncertainty."
#     })
#     return plots

# def generate_umap_plot():
#     def plot_fn():
#         plt.figure()
#         for i, label in enumerate(['LHS', 'Random']):
#             data = np.random.normal(loc=i*2, scale=0.7, size=(50,2))
#             plt.scatter(data[:,0], data[:,1], label=label, alpha=0.7)
#         plt.title("UMAP projection of input sampling")
#         plt.legend()
#         plt.xlabel("UMAP-1")
#         plt.ylabel("UMAP-2")
#     return plot_to_b64(plot_fn), "Data cluster structure suggests a Latin Hypercube Sampling (LHS) technique was used."

# def generate_error_histogram(y_true, y_pred, output_names):
#     plots = []
#     for i, name in enumerate(output_names):
#         def plot_fn():
#             plt.figure()
#             plt.hist(y_pred[:, i] - y_true[:, i], bins=20, alpha=0.8)
#             plt.xlabel("Prediction Error")
#             plt.ylabel("Frequency")
#             plt.title(f"Error Histogram ({name})")
#         plots.append({
#             "img_b64": plot_to_b64(plot_fn),
#             "title": f"Error Histogram", 
#             "caption": f"Histogram of prediction errors for {name}."
#         })
#     return plots


# ----- Surrogate Models with Uncertainty -----
class RandomForestWithUncertainty(RandomForestRegressor):
    def predict(self, X, return_std=False):
        mean = super().predict(X)
        if not return_std:
            return mean
        all_preds = np.stack([tree.predict(X) for tree in self.estimators_], axis=0)
        std = all_preds.std(axis=0)
        return mean, std

class GradientBoostingWithUncertainty(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.95, n_estimators=100):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.lower = GradientBoostingRegressor(loss="quantile", alpha=(1 - alpha) / 2, n_estimators=n_estimators)
        self.upper = GradientBoostingRegressor(loss="quantile", alpha=1 - (1 - alpha) / 2, n_estimators=n_estimators)
        self.mid = GradientBoostingRegressor(loss="squared_error", n_estimators=n_estimators)

    def fit(self, X, y):
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        self.mid.fit(X, y)
        return self

    def predict(self, X, return_std=False):
        y_pred = self.mid.predict(X)
        if not return_std:
            return y_pred
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        std = (upper - lower) / 2
        return y_pred, std

class KNeighborsRegressorWithUncertainty(KNeighborsRegressor):
    def predict(self, X, return_std=False):
        mean = super().predict(X)
        if not return_std:
            return mean
        neigh_ind = self.kneighbors(X, return_distance=False)
        y_neigh = self._y[neigh_ind]
        std = y_neigh.std(axis=1)
        return mean, std

class BootstrapLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_bootstraps=20):
        self.n_bootstraps = n_bootstraps

    def fit(self, X, y):
        self.models_ = []
        n = X.shape[0]
        for _ in range(self.n_bootstraps):
            idx = np.random.choice(n, n, replace=True)
            model = LinearRegression().fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict(self, X, return_std=False):
        all_preds = [m.predict(X).reshape(X.shape[0], -1) for m in self.models_]
        preds = np.stack(all_preds, axis=2)
        mean = preds.mean(axis=2)
        if not return_std:
            return mean
        std = preds.std(axis=2)
        return mean, std

class PerTargetRegressorWithStd(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        self.estimators = list(estimators)

    def fit(self, X, y):
        self.estimators_ = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for est, col in zip(self.estimators, y.T):
            est_fitted = clone(est)
            est_fitted.fit(X, col)
            self.estimators_.append(est_fitted)
        return self

    def predict(self, X, return_std=False):
        preds, stds = [], []
        for est in self.estimators_:
            if return_std:
                try:
                    pred, std = est.predict(X, return_std=True)
                except TypeError:
                    pred = est.predict(X)
                    std = np.full(pred.shape, np.nan)
                preds.append(pred.reshape(-1, 1))
                stds.append(std.reshape(-1, 1))
            else:
                pred = est.predict(X)
                preds.append(np.asarray(pred).reshape(-1, 1))
        if return_std:
            return np.hstack(preds), np.hstack(stds)
        return np.hstack(preds)


# ----- HTML Report Generation Wrapper -----
def generate_html_report(
    model_type, fidelity_levels, output_names, description, metrics,
    uncertainty_metrics, y_test, best_pred, best_std, best_model, X_train,
    n_train, n_test, cross_validation, seed, notes
):
    prediction_plots = generate_prediction_plot(y_test, best_pred, best_std, output_names)
    shap_plots = generate_shap_plot(best_model, X_train, output_names)
    # pdp_plots = generate_pdp_plot(output_names)
    pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=input_cols)
    uncertainty_plots = generate_uncertainty_plots()
    sampling_umap_plot, sampling_method_explanation = generate_umap_plot()
    other_plots = generate_error_histogram(y_test, best_pred, output_names)
    sampling_other_plots = []
    

    template_path = os.path.join(os.path.dirname(__file__), "../report/template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()
    template = Template(template_text)
    rendered = template.render(
        project_title="Multi-Fidelity Surrogate Modeling Report",
        model_type=model_type,
        fidelity_levels=fidelity_levels,
        output_names=output_names,
        description=description,
        metrics=metrics,
        uncertainty_metrics=uncertainty_metrics,
        uncertainty_plots=uncertainty_plots,
        prediction_plots=prediction_plots,
        shap_plots=shap_plots,
        pdp_plots=pdp_plots,
        sampling_umap_plot=sampling_umap_plot,
        sampling_method_explanation=sampling_method_explanation,
        sampling_other_plots=sampling_other_plots,
        other_plots=other_plots,
        n_train=n_train,
        n_test=n_test,
        cross_validation=cross_validation,
        seed=seed,
        notes=notes,
    )
    return rendered

# ----- Streamlit App -----
st.title("Multi-Output Surrogate Model Grid Search & Report Generator")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview of Data:", df.head())

    with st.form("column_selection"):
        input_cols = st.multiselect("Select input features", options=df.columns)
        output_cols = st.multiselect("Select output targets", options=df.columns)
        description = st.text_area("Optional: Project description")
        submitted = st.form_submit_button("Run Grid Search")

    if submitted and input_cols and output_cols:
        X = df[input_cols].values
        y = df[output_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

        surrogate_defs = [
            ("gpr", GaussianProcessRegressor, {"alpha": [1e-4], "kernel": [RBF(), Matern(nu=1.5)]}),
            ("rf", RandomForestWithUncertainty, {"n_estimators": [50], "max_depth": [3, None]}),
            ("gb", GradientBoostingWithUncertainty, {"alpha": [0.95], "n_estimators": [50]}),
            ("knn", KNeighborsRegressorWithUncertainty, {"n_neighbors": [3]}),
            ("blr", BootstrapLinearRegression, {"n_bootstraps": [20]}),
        ]

        configs = [(name, Est, params) for name, Est, grid in surrogate_defs for params in ParameterGrid(grid)]

        best_score = np.inf
        best_combo = None
        best_pred = None
        best_std = None
        best_model = None

        for est_params in ParameterGrid({"combos": [configs]*y_train.shape[1]}):
            try:
                combo = est_params["combos"][:y_train.shape[1]]
                estimators = [est(**params) for (_, est, params) in combo]
                model = PerTargetRegressorWithStd(estimators)
                model.fit(X_train, y_train)
                pred, std = model.predict(X_test, return_std=True)
                score = mean_squared_error(y_test, pred)
                if score < best_score:
                    best_score = score
                    best_combo = [dict(model=name, **params) for (name, _, params) in combo]
                    best_pred, best_std = pred, std
                    best_model = model
            except Exception:
                continue

        st.write("### Best Model Configuration")
        st.json(best_combo)

        st.write("### Metrics Table")
        metrics = {}
        for i, name in enumerate(output_cols):
            y_true = y_test[:, i]
            y_pred = best_pred[:, i]
            metrics[name] = {
                "r2": r2_score(y_true, y_pred),
                "rmse": mean_squared_error(y_true, y_pred, squared=False),
                "mae": mean_absolute_error(y_true, y_pred),
                "mean_predicted_std": float(np.mean(best_std[:, i]))
            }
        st.dataframe(pd.DataFrame(metrics).T)

        html = generate_html_report(
            model_type="PerTargetRegressorWithStd",
            fidelity_levels=[],
            output_names=output_cols,
            description=description,
            metrics=metrics,
            uncertainty_metrics={"dummy_metric": 0.0},
            y_test=y_test,
            best_pred=best_pred,
            best_std=best_std,
            best_model=best_model,
            X_train=X_train,
            n_train=X_train.shape[0],
            n_test=X_test.shape[0],
            cross_validation="None",
            seed=0,
            notes="Streamlit-generated report."
        )

        st.download_button("Download HTML Report", html, file_name="model_report.html", mime="text/html")