# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from jinja2 import Template
import joblib

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
from multioutreg.model_selection import AutoDetectMultiOutputRegressor

# Multi-objective regret imports
from multioutreg.metrics import hypervolume_regret, scalarized_regret, epsilon_regret


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
    template_path: Optional[Union[str, os.PathLike]] = None,  # For unit tests
    shap_plot: str | None = None,
) -> str:
    """
    Generate an HTML report summarizing surrogate model results, including performance metrics,
    uncertainty plots, SHAP values, PDPs, PCA visualizations, and residual analysis.
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

    prediction_plots = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi,
        best_pred,
        best_std,
        y_test,
        target_list=output_names,
    )
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

    unc_img = safe_plot_b64(
        plot_coverage, y_test, best_pred, best_std, output_names=output_names
    )
    uncertainty_plots = [
        {
            "img_b64": unc_img,
            "title": "Coverage Plot",
            "caption": "Nominal vs empirical coverage.",
        }
    ]

    pdp_plots = generate_pdp_plot(best_model, X_train, output_names, feature_names=feature_names)

    sampling_umap_plot, sampling_method_explanation = generate_umap_plot(X_train)

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
        # Ensure y is 2-D for single-output selections
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        feature_names = list(input_cols)
        used_feature_names = feature_names

        # PCA
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

        # Ensure predictions/uncertainty are 2-D even for single-output models
        if isinstance(best_pred, list):
            best_pred = np.asarray(best_pred)
        if isinstance(best_std, list):
            best_std = np.asarray(best_std)
        if best_pred.ndim == 1:
            best_pred = best_pred.reshape(-1, 1)
        if best_std.ndim == 1:
            best_std = best_std.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        best_model = model
        best_combo = [m.__class__.__name__ for m in model.models_]

        st.write("### Selected Surrogates")
        st.json(best_combo)

        st.write("### Metrics Table")
        metrics = {}
        for i, name in enumerate(output_cols):
            y_true_i = y_test[:, i]
            y_pred_i = best_pred[:, i]
            metrics[name] = {
                "r2": r2_score(y_true_i, y_pred_i),
                "rmse": mean_squared_error(y_true_i, y_pred_i, squared=False),
                "mae": mean_absolute_error(y_true_i, y_pred_i),
                "mean_predicted_std": float(np.mean(best_std[:, i])),
            }
        st.dataframe(pd.DataFrame(metrics).T)

        # ----- Multi-Objective Regret (robust to m=1 or m>1) -----
        Y_true = y_test     # (n, m)
        Y_pred = best_pred  # (n, m)
        m = Y_true.shape[1]

        # Choose weights for scalarized regret
        if m == 1:
            W = 1.0
        elif m == 2:
            W = np.array([
                [0.5, 0.5],
                [0.2, 0.8],
                [0.8, 0.2],
            ])
        else:
            rng = np.random.default_rng(7)
            W = rng.dirichlet(alpha=np.ones(m), size=8)  # (8, m)

        r_hv = hypervolume_regret(Y_true, Y_pred)
        r_sc = scalarized_regret(Y_true, Y_pred, weights=W, reduce="mean")
        r_eps = epsilon_regret(Y_true, Y_pred)

        st.subheader("Multi-Objective Regret")
        st.write(f"Hypervolume regret: {r_hv:.4g}")
        st.write(f"Scalarized regret (mean): {r_sc:.4g}")
        st.write(f"Epsilon regret: {r_eps:.4g}")

        # SHAP plots
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

        # ----- Downloads -----
        # A) HTML report
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
        )
        st.download_button(
            "Download HTML Report",
            html,
            file_name="model_report_auto.html",
            mime="text/html",
        )

        # B) Trained model as .joblib
        model_buf = io.BytesIO()
        joblib.dump(best_model, model_buf)
        model_buf.seek(0)
        st.download_button(
            "Download Trained Model (.joblib)",
            data=model_buf.getvalue(),
            file_name="multioutreg_best_model.joblib",
            mime="application/octet-stream",
        )
