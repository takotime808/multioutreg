# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Union, Any
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from jinja2 import Template


# from multioutreg.gui.report_plotting_utils import (
#     # plot_to_b64,
#     generate_prediction_plot,
#     generate_shap_plot,
#     generate_pdp_plot,
#     generate_uncertainty_plots,
#     generate_umap_plot,
#     generate_error_histogram
# )
from multioutreg.gui.report_plotting_utils import (
    generate_shap_plot,
    generate_error_histogram,
    generate_pca_variance_plot,
)
from multioutreg.figures.doe_plots import make_doe_plot
from multioutreg.figures.model_comparison import plot_surrogate_model_summary
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
# from multioutreg.figures.shap_multioutput import plot_multioutput_shap_bar_subplots
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
# from multioutreg.figures.prediction_plots import plot_predictions
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi

# ----- Surrogate Models with Uncertainty -----
class RandomForestWithUncertainty(RandomForestRegressor):
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the random forest, optionally returning standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return the standard deviation of predictions.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingWithUncertainty":
        """
        Fit the quantile and mid regressors.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        GradientBoostingWithUncertainty
            The fitted model.
        """
        self.lower.fit(X, y)
        self.upper.fit(X, y)
        self.mid.fit(X, y)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the model.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
        y_pred = self.mid.predict(X)
        if not return_std:
            return y_pred
        lower = self.lower.predict(X)
        upper = self.upper.predict(X)
        std = (upper - lower) / 2
        return y_pred, std


class KNeighborsRegressorWithUncertainty(KNeighborsRegressor):
    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using KNN, optionally returning standard deviation.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BootstrapLinearRegression":
        """
        Fit bootstrapped linear regression models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target array.

        Returns
        -------
        BootstrapLinearRegression
            The fitted model.
        """
        self.models_ = []
        n = X.shape[0]
        for _ in range(self.n_bootstraps):
            idx = np.random.choice(n, n, replace=True)
            model = LinearRegression().fit(X[idx], y[idx])
            self.models_.append(model)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using ensemble of linear models.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerTargetRegressorWithStd":
        """
        Fit separate estimators for each output column.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Multi-output target matrix.

        Returns
        -------
        PerTargetRegressorWithStd
            The fitted model.
        """
        self.estimators_ = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        for est, col in zip(self.estimators, y.T):
            est_fitted = clone(est)
            est_fitted.fit(X, col)
            self.estimators_.append(est_fitted)
        return self

    def predict(
            self,
            X: np.ndarray,
            return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using individual estimators for each target dimension.

        Parameters
        ----------
        X : np.ndarray
            Input feature array.
        return_std : bool, optional
            Whether to return standard deviation.

        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Mean predictions or (mean, std) tuple.
        """
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
    model_type: str,
    fidelity_levels: List[str],
    output_names: List[str],
    description: str,
    metrics: Dict[str, Dict[str, float]],
    uncertainty_metrics: Dict[str, float],
    y_test: np.ndarray,
    best_pred: np.ndarray,
    best_std: np.ndarray,
    best_model: Any,
    X_train: np.ndarray,
    n_train: int,
    n_test: int,
    cross_validation: str,
    seed: int,
    notes: str,
    feature_names_pca: List[str] | None = None,
    pca_explained_variance: List[float] | None = None,
    pca_variance_plot: str | None = None,
) -> str:
    """
    Generate an HTML report from model training and evaluation results.

    Parameters
    ----------
    model_type : str
        Type of model used.
    fidelity_levels : List[str]
        Fidelity levels used (if any).
    output_names : List[str]
        Names of the output dimensions.
    description : str
        Description of the modeling project.
    metrics : Dict[str, Dict[str, float]]
        Dictionary of evaluation metrics for each output.
    uncertainty_metrics : Dict[str, float]
        Dictionary of uncertainty metrics.
    y_test : np.ndarray
        Ground truth values for test set.
    best_pred : np.ndarray
        Predictions from the best model.
    best_std : np.ndarray
        Uncertainty estimates from the best model.
    best_model : Any
        The best performing model.
    X_train : np.ndarray
        Training input features.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    cross_validation : str
        Description of the CV strategy.
    seed : int
        Random seed used.
    notes : str
        Additional notes.
    feature_names_pca : List[str] | None, optional
        Names to use for PCA.
    pca_explained_variance : List[float] | None, optional
        Explained variance ratios from PCA if applied.
    pca_variance_plot : str | None, optional
        Base64-encoded plot showing PCA variance ratios.

    Returns
    -------
    str
        Rendered HTML report.
    """
    # prediction_plots = generate_prediction_plot(y_test, best_pred, best_std, output_names)
    # # pdp_plots = generate_pdp_plot(output_names)
    # pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=input_cols)
    # uncertainty_plots = generate_uncertainty_plots()

    prediction_plots = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi,
        best_pred,
        best_std,
        y_test,
        # max_cols=3,
        target_list=output_names,
    )
    # # PREDICTION PLOTS OPTION 2
    # prediction_plots = {}
    for i, name in enumerate(output_names):
        prediction_plots[name] = safe_plot_b64(
            plot_predictions_with_error_bars,
            y_test[:, [i]],
            best_pred[:, [i]],
            best_std[:, [i]],
            output_names=[name],
            n_cols=3
        )

    shap_plots = generate_shap_plot(best_model, X_train, output_names)
    # shap_img = safe_plot_b64(
    #     plot_multioutput_shap_bar_subplots,
    #     best_model, X_train,
    #     feature_names=input_cols, output_names=output_names
    # )
    # shap_plots = {name: shap_img for name in output_names}

    unc_img = safe_plot_b64(
        plot_coverage, y_test, best_pred, best_std, output_names=output_names
    )
    # uncertainty_plots = [{"img_b64": unc_img, "title": "Coverage Plot", "caption": "Nominal vs empirical coverage."}]
    uncertainty_plots = [
        {
            "img_b64": unc_img,
            "title": "Coverage Plot",
            "caption": "Nominal vs empirical coverage.",
        }
    ]

    pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=feature_names)
    
    sampling_umap_plot, sampling_method_explanation = generate_umap_plot(X_train)
    # other_plots = generate_error_histogram(y_test, best_pred, output_names)
    other_img = safe_plot_b64(
        plot_residuals_multioutput_with_regplot,
        best_pred,
        y_test,
        target_list=output_names,
    )
    sampling_other_plots = [
        {
            "img_b64": other_img,
            "title": "Residuals",
            "caption": "Residual vs predicted values.",
        }
    ]

    other_plots = generate_error_histogram(y_test, best_pred, output_names)

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
        feature_names_pca=feature_names_pca,
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
    )
    return rendered


# ----- Streamlit App -----
st.title("Multi-Output Surrogate Model Grid Search & Report Generator")

uploaded_file = st.file_uploader(
    "Upload CSV file. Example files can be found in the repo: `multioutreg/docs/_static/example_datasets/`.",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview of Data:", df.head())

    with st.form("column_selection"):
        input_cols = st.multiselect("Select input features", options=df.columns)
        output_cols = st.multiselect("Select output targets", options=df.columns)
        use_pca = st.checkbox("Apply PCA to input features")
        n_components = None
        if use_pca:
            # max_comp = max(1, len(input_cols)) if input_cols else len(df.columns)
            max_comp = len(df.columns)
            n_components = st.number_input(
                "Number of PCA components",
                min_value=1,
                max_value=max_comp,
                value=min(2, max_comp),
                step=1,
            )
        description = st.text_area("Optional: Project description")
        submitted = st.form_submit_button("Run Grid Search")

    if submitted and input_cols and output_cols:
        X = df[input_cols].values
        y = df[output_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        feature_names = list(input_cols)

        # # Show seaborn PairGrid plot with KDE in lower triangle
        # if df.shape[1] >= 2:
        #     st.write("### PairGrid Visualization (KDE Lower Triangle)")

        #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        #     if len(numeric_cols) >= 2:
        #         g = make_doe_plot(df=df, numeric_cols=numeric_cols)
        #         st.pyplot(g.fig)
        #     else:
        #         st.info("Not enough numeric columns for pairwise plot.")

        # PCA.
        pca_variance_plot = None
        pca_explained_variance = None
        if use_pca:
            pca = PCA(n_components=int(n_components))
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            feature_names_pca = [f"PC{i+1}" for i in range(int(n_components))]
            # input_cols = feature_names
            pca_variance_plot = generate_pca_variance_plot(pca)
            pca_explained_variance = pca.explained_variance_ratio_.tolist()

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

        for est_params in ParameterGrid({"combos": [configs] * y_train.shape[1]}):
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

        # # Show seaborn PairGrid plot with KDE in lower triangle
        # if df.shape[1] >= 2:
        #     st.write("### PairGrid Visualization (KDE Lower Triangle)")

        #     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        #     if len(numeric_cols) >= 2:
        #         g = make_doe_plot(df=df, numeric_cols=numeric_cols)
        #         st.pyplot(g.fig)
        #         del g
        #         grid_plot = plot_surrogate_model_summary(
        #                 X_train=X_train,
        #                 X_test=X_test,
        #                 Y_train=y_train,
        #                 Y_test=y_test,
        #                 model=best_model,
        #                 savefig=False,
        #         )
        #         st.pyplot(grid_plot)
        #         del grid_plot
        #     else:
        #         st.info("Not enough numeric columns for pairwise plot.")

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
                "mean_predicted_std": float(np.mean(best_std[:, i])),
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
            notes="Generated report.",
            pca_explained_variance=pca_explained_variance,
            pca_variance_plot=pca_variance_plot,
        )

        st.download_button("Download HTML Report", html, file_name="model_report.html", mime="text/html")