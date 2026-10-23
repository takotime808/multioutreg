# Copyright (c) 2025 takotime808

"""HTML report generation for AutoDetectMultiOutputRegressor results.

Shared by the Streamlit page and the ``grid_search_auto_detect`` CLI command.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Union
from jinja2 import Template

from multioutreg.figures.error_histograms import generate_error_histogram
from multioutreg.figures.shap_multioutput import (
    generate_shap_plot,
    plot_multioutput_shap_bar_subplots,
)
from multioutreg.figures.pca_plots import generate_pca_variance_plot
from multioutreg.figures.pdp_plots import generate_pdp_plot
from multioutreg.utils.figure_utils import safe_plot_b64
from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.figures.prediction_plots import plot_predictions_with_error_bars
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals_ordered,
    plot_conformal_vs_gaussian,
)


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
    feature_names: List[str] | None = None,
    feature_names_pca: List[str] | None = None,
    pca_explained_variance: List[float] | None = None,
    pca_variance_plot: str | None = None,
    pca_method: str | None = None,
    pca_threshold: float | None = None,
    pca_n_components: int | None = None,
    kaiser_rule_suggestion: str | None = None,
    template_path: Optional[Union[str, os.PathLike]] = None, # For unit tests
    shap_plot: str | None = None,
    conformal_intervals: tuple | None = None,
    conformal_alpha: float | None = None,
    calibration_residuals: np.ndarray | None = None,
    imputation_summary: dict | None = None,
    regret_metrics: Dict[str, float] | None = None,
    regret_bounds: Dict[str, Optional[float]] | None = None,
    regret_settings: Dict[str, Any] | None = None,
) -> str:
    """
    Generate an HTML report summarizing surrogate model results, including performance metrics,
    uncertainty plots, SHAP values, PDPs, PCA visualizations, and residual analysis.

    This function wraps model output and diagnostics into a styled report by rendering a Jinja2
    HTML template. It supports optional PCA annotations, sampling visualizations, and error histograms.
    Designed to be used interactively or programmatically within the Streamlit grid search app.

    Parameters
    ----------
    model_type : str
        Type of surrogate model used (e.g., "AutoDetectMultiOutputRegressor").
    fidelity_levels : List[str]
        List of fidelity levels (e.g., ["Low", "High"]) if multi-fidelity data is involved.
    output_names : List[str]
        Names of output variables for multi-output regression.
    description : str
        Optional project description to embed in the report.
    metrics : Dict[str, Dict[str, float]]
        Dictionary containing performance metrics (r2, rmse, mae, etc.) for each output.
    uncertainty_metrics : Dict[str, float]
        Global uncertainty metrics across all outputs.
    y_test : np.ndarray
        Ground truth test targets.
    best_pred : np.ndarray
        Model predictions on test data.
    best_std : np.ndarray
        Predicted standard deviation (uncertainty) per test prediction.
    best_model : Any
        Trained surrogate model object.
    X_train : np.ndarray
        Training feature matrix.
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    cross_validation : str
        Text description of the cross-validation strategy used.
    seed : int
        Random seed used during data splitting or model training.
    notes : str
        Additional notes to be included in the report.
    feature_names : List[str] | None, optional
        Names of original input features (used in PDP plots).
    feature_names_pca : List[str] | None, optional
        Names of PCA components (used for visualization if PCA was applied).
    pca_explained_variance : List[float] | None, optional
        Explained variance ratios from PCA analysis.
    pca_variance_plot : str | None, optional
        Base64-encoded PNG string of the scree plot (PCA variance).
    pca_method : str | None, optional
        Method used to select the number of PCA components ("Manual", "Kaiser rule", etc.).
    pca_threshold : float | None, optional
        Explained variance threshold used, if applicable.
    pca_n_components : int | None, optional
        Number of PCA components retained.
    kaiser_rule_suggestion : str | None, optional
        Human-readable explanation of the Kaiser rule component count.
    template_path : str | None, optional
        Path to the HTML Jinja2 template. Used for unit testing or template overrides.
        If not provided, falls back to the default report template or env variable "MOR_TEMPLATE_PATH".
    shap_plot: str | None,
        Generate multioupyt shap plots as subplots on one figure.
    conformal_intervals : tuple | None, optional
        ``(y_lower, y_upper)`` conformal intervals on the test set.
    conformal_alpha : float | None, optional
        Miscoverage level of ``conformal_intervals``.
    calibration_residuals : np.ndarray | None, optional
        ``|y_cal - y_cal_pred|`` from the held-out conformal calibration set.
        The Gaussian-vs-conformal coverage plot is drawn only when given.
    regret_metrics : Dict[str, float] | None, optional
        Multi-objective regret values keyed ``"hv"``, ``"scalar"`` and ``"eps"``
        (see :mod:`multioutreg.metrics`). The regret section renders only when given.
    regret_bounds : Dict[str, float | None] | None, optional
        Optional max-acceptable value per regret key, shown as PASS/FAIL.
    regret_settings : Dict[str, Any] | None, optional
        Settings used to compute the regret (reference point, weights mode, ...).

    Returns
    -------
    str
        Fully rendered HTML report as a string, ready to be saved or displayed.
    """
    DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "template.html")
    
    # For unit tests
    if template_path is None:
        template_path = os.getenv("MOR_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH)

    # Check that the template exists
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

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

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    if shap_plot is None:
        shap_plot = safe_plot_b64(
            plot_multioutput_shap_bar_subplots,
            best_model,
            X_train,
            feature_names=feature_names,
            output_names=output_names,
        )
    shap_plots = {"SHAP Summary": shap_plot}

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

    # Conformal prediction plots
    conformal_plots_list = []
    if conformal_intervals is not None and conformal_alpha is not None:
        y_lower, y_upper = conformal_intervals
        alpha = conformal_alpha

        conformal_ordered_img = safe_plot_b64(
            plot_conformal_intervals_ordered,
            y_test, y_lower, y_upper,
            y_pred=best_pred,
            output_names=output_names,
            alpha=alpha,
        )
        conformal_plots_list.append({
            "img_b64": conformal_ordered_img,
            "title": f"Conformal Prediction Intervals ({int((1-alpha)*100)}%)",
            "caption": "Distribution-free prediction intervals ordered by observed value. Guaranteed marginal coverage.",
        })

        # Conformal curves must use calibration residuals; test residuals
        # would match the diagonal by construction.
        if calibration_residuals is not None:
            conformal_vs_gauss_img = safe_plot_b64(
                plot_conformal_vs_gaussian,
                y_test, best_pred, best_std,
                calibration_residuals,
                output_names=output_names,
            )
            conformal_plots_list.append({
                "img_b64": conformal_vs_gauss_img,
                "title": "Gaussian CI vs Conformal PI Coverage",
                "caption": "Comparison of nominal vs empirical coverage for Gaussian confidence intervals and conformal prediction intervals.",
            })

    # template_path = os.path.join(os.path.dirname(__file__), "../../report/template.html")
    # with open(template_path, "r", encoding="utf-8") as f:
    #     template_text = f.read()

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    template = Template(template_text)

    rendered = template.render(
        project_title="Auto-Detected Multi-Fidelity Surrogate Modeling Report",
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
        conformal_plots=conformal_plots_list,
        n_train=n_train,
        n_test=n_test,
        cross_validation=cross_validation,
        seed=seed,
        notes=notes,
        feature_names_pca=feature_names_pca,
        pca_explained_variance=pca_explained_variance,
        pca_variance_plot=pca_variance_plot,
        pca_method=pca_method,
        pca_threshold=pca_threshold,
        pca_n_components=pca_n_components,
        kaiser_rule_suggestion=kaiser_rule_suggestion,
        imputation_summary=imputation_summary,
        regret_metrics=regret_metrics,
        regret_bounds=regret_bounds,
        regret_settings=regret_settings,
    )
    return rendered
