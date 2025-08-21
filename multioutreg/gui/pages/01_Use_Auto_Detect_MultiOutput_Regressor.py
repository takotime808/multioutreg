# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from jinja2 import Template
from sklearn.linear_model import LinearRegression
from multioutreg.timeseries import TimeSeriesForecaster


# from multioutreg.gui.report_plotting_utils import (
#     # plot_to_b64,
#     generate_prediction_plot,
#     generate_shap_plot,
#     generate_pdp_plot,
#     generate_uncertainty_plots,
#     generate_umap_plot,
#     generate_error_histogram
#     generate_pca_variance_plot,
# )

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
# from multioutreg.figures.shap_multioutput import plot_multioutput_shap_bar_subplots
from multioutreg.figures.coverage_plots import plot_coverage
from multioutreg.figures.residuals import plot_residuals_multioutput_with_regplot
# from multioutreg.figures.prediction_plots import plot_predictions
from multioutreg.figures.confidence_intervals import plot_intervals_ordered_multi
from multioutreg.model_selection import AutoDetectMultiOutputRegressor

# # NOTE: NOT used...yet.
# from multioutreg.figures.doe_plots import make_doe_plot
# from multioutreg.figures.model_comparison import plot_surrogate_model_summary



# ----- Time-Series Forecasting Helper -----
def forecast_series(
    series: np.ndarray,
    lags: int,
    horizon: int,
    base_estimator: Any | None = None,
) -> np.ndarray:
    """Forecast a univariate series using :class:`TimeSeriesForecaster`.

    Parameters
    ----------
    series:
        Historical data points.
    lags:
        Number of lagged observations to use.
    horizon:
        Number of future steps to predict.
    base_estimator:
        Optional base regressor. Defaults to
        :class:`~sklearn.linear_model.LinearRegression`.

    Returns
    -------
    np.ndarray
        Forecasted values of length ``horizon``.
    """
    if base_estimator is None:
        base_estimator = LinearRegression()
    forecaster = TimeSeriesForecaster(base_estimator, lags=lags, horizon=horizon)
    forecaster.fit(series)
    return forecaster.predict(series)


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
    ts_forecast: np.ndarray | None = None,
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
    ts_forecast : np.ndarray | None, optional
        Optional time-series forecast to embed in the report.

    Returns
    -------
    str
        Fully rendered HTML report as a string, ready to be saved or displayed.
    """
    DEFAULT_TEMPLATE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../report/template.html"
    )
    
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
    
    # template_path = os.path.join(os.path.dirname(__file__), "../../report/template.html")
    # with open(template_path, "r", encoding="utf-8") as f:
    #     template_text = f.read()

    with open(DEFAULT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
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
        ts_forecast=ts_forecast,
    )
    return rendered


# ----- Streamlit App -----
st.title("Auto-Detected Multi-Output Surrogate Model Grid Search & Report Generator")

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
        pca_method = None
        pca_threshold = None
        if use_pca:
            # # max_comp = max(1, len(input_cols)) if input_cols else len(df.columns)
            # max_comp = len(df.columns)
            # n_components = st.number_input(
            #     "Number of PCA components",
            #     min_value=1,
            #     max_value=max_comp,
            #     value=min(2, max_comp),
            #     step=1,
            # )
            max_comp = len(df.columns)
            pca_method = st.selectbox(
                "PCA component selection method",
                ["Manual", "Explained variance threshold", "Kaiser rule"],
            )
            if pca_method == "Manual":
                n_components = st.number_input(
                    "Number of PCA components",
                    min_value=1,
                    max_value=max_comp,
                    value=min(2, max_comp),
                    step=1,
                )
            elif pca_method == "Explained variance threshold":
                pca_threshold = st.slider(
                    "Explained variance threshold",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.9,
                    step=0.01,
                )

        description = st.text_area("Optional: Project description")
        submitted = st.form_submit_button("Run Grid Search")

    if submitted and input_cols and output_cols:
        X = df[input_cols].values
        y = df[output_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        feature_names = list(input_cols)
        used_feature_names = feature_names

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
        pca_n_components = None
        feature_names_pca = None
        kaiser_rule_suggestion = None
        if use_pca:
            preview_pca = PCA().fit(X_train)
            kaiser_k = int(np.sum(preview_pca.explained_variance_ > 1))
            if pca_method == "Manual":
                pca_n_components = int(n_components)
            elif pca_method == "Explained variance threshold":
                cum = np.cumsum(preview_pca.explained_variance_ratio_)
                pca_n_components = int(np.searchsorted(cum, pca_threshold) + 1)
            else:  # Kaiser rule
                pca_n_components = max(1, kaiser_k)
            pca = PCA(n_components=pca_n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            feature_names_pca = [f"PC{i+1}" for i in range(pca_n_components)]
            pca_variance_plot = generate_pca_variance_plot(
                preview_pca,
                n_selected=pca_n_components,
                threshold=pca_threshold if pca_method == "Explained variance threshold" else None,
            )
            pca_explained_variance = preview_pca.explained_variance_ratio_.tolist()
            kaiser_rule_suggestion = f"Kaiser rule suggests **{kaiser_k}** components (eigenvalues > 1)."
            st.markdown(kaiser_rule_suggestion)
            st.image(
                f"data:image/png;base64,{pca_variance_plot}",
                caption="Scree Plot",
                use_column_width=True,
            )
            used_feature_names = feature_names_pca

        model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
        model.fit(X_train, y_train)
        best_pred, best_std = model.predict(X_test, return_std=True)
        best_model = model
        best_combo = [m.__class__.__name__ for m in model.models_]

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

        st.write("### Selected Surrogates")
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

        # Shap plots
        shap_img = safe_plot_b64(
            plot_multioutput_shap_bar_subplots,
            best_model,
            X_train,
            feature_names=used_feature_names,
            output_names=output_cols,
        )
        st.write("### SHAP Summary Plot")
        st.image(
            f"data:image/png;base64,{shap_img}",
            caption="Mean(|SHAP value|) for each feature and output",
            use_column_width=True,
        )
        ts_forecast = None
        try:
            ts_forecast = forecast_series(y_train[:, 0], lags=3, horizon=2)
        except Exception:
            ts_forecast = None

        html = generate_html_report(
            model_type="AutoDetectMultiOutputRegressor",
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
            feature_names=used_feature_names,
            pca_explained_variance=pca_explained_variance,
            pca_variance_plot=pca_variance_plot,
            pca_method=pca_method,
            pca_threshold=pca_threshold,
            pca_n_components=pca_n_components,
            kaiser_rule_suggestion=kaiser_rule_suggestion,
            shap_plot=shap_img,
            ts_forecast=ts_forecast,
        )

        st.download_button("Download HTML Report", html, file_name="model_report_auto.html", mime="text/html")