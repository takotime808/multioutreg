# Copyright (c) 2025 takotime808

"""Streamlit script performing a grid search over several surrogate models."""

import os
import io
import re
import hashlib
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


# ---------- session state init ----------
if "artifacts" not in st.session_state:
    st.session_state.artifacts = {
        "last_run_id": None,
        "model_bytes": None,
        "html_report": None,
        "metrics_df": None,
        "best_combo": None,
        "y_test": None,
        "best_pred": None,
        "best_std": None,
        "used_feature_names": None,
        "output_cols": None,
        "settings": None,
        "regret": {
            "hv": None,
            "scalar": None,
            "eps": None,
            "bounds": {"hv": None, "scalar": None, "eps": None},
        },
    }


# ---------- helpers ----------
def _serialize_model(model: Any) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.getvalue()


def _config_hash(
    file_bytes: bytes,
    input_cols: List[str],
    output_cols: List[str],
    use_pca: bool,
    pca_method: Optional[str],
    pca_threshold: Optional[float],
    n_components: Optional[int],
    hv_ref_mode: str,
    hv_margin_pct: Optional[float],
    hv_ref_custom: Optional[str],
    weights_mode: str,
    weights_custom: Optional[str],
    bounds_hv: Optional[str],
    bounds_scalar: Optional[str],
    bounds_eps: Optional[str],
    test_size: float = 0.25,
    random_state: int = 0,
) -> str:
    h = hashlib.sha1()
    h.update(file_bytes)
    h.update(("|".join(input_cols)).encode("utf-8"))
    h.update(("|".join(output_cols)).encode("utf-8"))
    for v in [
        use_pca,
        pca_method,
        pca_threshold,
        n_components,
        hv_ref_mode,
        hv_margin_pct,
        hv_ref_custom,
        weights_mode,
        weights_custom,
        bounds_hv,
        bounds_scalar,
        bounds_eps,
        test_size,
        random_state,
    ]:
        h.update(str(v).encode() if v is not None else b"")
    return h.hexdigest()


def _parse_optional_float(txt: Optional[str]) -> Optional[float]:
    if not txt:
        return None
    try:
        return float(txt.strip())
    except Exception:
        return None


def _parse_csv_floats(txt: str) -> List[float]:
    return [float(x) for x in re.split(r"[,\s]+", txt.strip()) if x]


def _parse_weight_matrix(txt: str, m: int) -> Optional[np.ndarray]:
    """
    Each line = one weight vector (comma/space separated). Returns (k,m) or None if invalid.
    """
    if not txt:
        return None
    rows = []
    for line in txt.strip().splitlines():
        if not line.strip():
            continue
        vals = _parse_csv_floats(line)
        if len(vals) != m:
            return None
        rows.append(vals)
    if not rows:
        return None
    W = np.asarray(rows, dtype=float)
    # Normalize rows to simplex robustly
    W = np.clip(W, 0.0, np.inf)
    sums = W.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return W / sums


def _auto_reference_point(Y_true: np.ndarray, Y_pred: np.ndarray, margin_pct: float) -> np.ndarray:
    """
    Compute a safe minimization reference point:
    worst + margin_pct% * span, with a 1.0 fallback when span==0.
    """
    allY = np.vstack([Y_true, Y_pred])
    worst = np.max(allY, axis=0)
    span = np.ptp(allY, axis=0)
    return worst + np.where(span > 0, (margin_pct / 100.0) * span, 1.0)


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
    DEFAULT_TEMPLATE_PATH = os.path.join(
        os.path.dirname(__file__),
        "../../report/template.html",
    )
    if template_path is None:
        template_path = os.getenv("MOR_TEMPLATE_PATH", DEFAULT_TEMPLATE_PATH)
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")

    prediction_plots = {}
    prediction_plots["all_in_one"] = safe_plot_b64(
        plot_intervals_ordered_multi, best_pred, best_std, y_test, target_list=output_names
    )
    for i, name in enumerate(output_names):
        prediction_plots[name] = safe_plot_b64(
            plot_predictions_with_error_bars,
            y_test[:, [i]],
            best_pred[:, [i]],
            best_std[:, [i]],
            output_names=[name],
            n_cols=3,
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
        plot_residuals_multioutput_with_regplot, best_pred, y_test, target_list=output_names
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
    type=["csv"],
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("## Preview of Data:", df.head())

    with st.form("column_selection", clear_on_submit=False):
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

        # ------------------- Regret settings & bounds -------------------
        st.markdown("### Regret settings & bounds")
        colA, colB = st.columns(2)
        with colA:
            hv_ref_mode = st.selectbox(
                "Hypervolume reference point",
                ["Auto (10% beyond worst)", "Auto with margin (%)", "Custom"],
                index=0,
            )
            hv_margin_pct = None
            hv_ref_custom = None
            if hv_ref_mode == "Auto with margin (%)":
                hv_margin_pct = st.number_input(
                    "Reference margin (%)", 0.0, 100.0, value=10.0, step=0.5
                )
            elif hv_ref_mode == "Auto (10% beyond worst)":
                hv_margin_pct = 10.0
            else:
                hv_ref_custom = st.text_input(
                    "Custom reference point (comma/space separated; one value per objective)",
                    placeholder="e.g., 3.0, 3.0",
                )
        with colB:
            weights_mode = st.selectbox(
                "Scalarized weights",
                ["Auto (Dirichlet k=8)", "Uniform", "Custom"],
                index=0,
            )
            weights_custom = None
            if weights_mode == "Custom":
                weights_custom = st.text_area(
                    "Custom weight vectors (one line per vector; comma/space separated)",
                    placeholder="e.g.\n0.5, 0.5\n0.2, 0.8\n0.8, 0.2",
                    height=120,
                )

        st.markdown("#### Optional pass/fail bounds (leave blank to skip)")
        col1, col2, col3 = st.columns(3)
        with col1:
            bounds_hv = st.text_input("Max HV regret (≤)", value="")
        with col2:
            bounds_scalar = st.text_input("Max scalarized regret (≤)", value="")
        with col3:
            bounds_eps = st.text_input("Max ε-regret (≤)", value="")
        # ---------------------------------------------------------------

        description = st.text_area("Optional: Project description")
        submitted = st.form_submit_button("Run Grid Search")

    # ---------- recompute only on submit ----------
    if submitted and input_cols and output_cols:
        file_bytes = uploaded_file.getbuffer()
        run_id = _config_hash(
            file_bytes=file_bytes,
            input_cols=list(input_cols),
            output_cols=list(output_cols),
            use_pca=use_pca,
            pca_method=pca_method,
            pca_threshold=pca_threshold,
            n_components=int(n_components) if (use_pca and pca_method == "Manual") else None,
            hv_ref_mode=hv_ref_mode,
            hv_margin_pct=hv_margin_pct,
            hv_ref_custom=hv_ref_custom,
            weights_mode=weights_mode,
            weights_custom=weights_custom,
            bounds_hv=bounds_hv,
            bounds_scalar=bounds_scalar,
            bounds_eps=bounds_eps,
        )

        # compute
        X = df[input_cols].values
        y = df[output_cols].values
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        feature_names = list(input_cols)
        used_feature_names = feature_names

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
            kaiser_rule_suggestion = (
                f"Kaiser rule suggests **{kaiser_k}** components (eigenvalues > 1)."
            )
            used_feature_names = feature_names_pca

        model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
        model.fit(X_train, y_train)
        best_pred, best_std = model.predict(X_test, return_std=True)

        # Ensure 2-D
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

        best_combo = [m.__class__.__name__ for m in model.models_]

        # Metrics table
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
        metrics_df = pd.DataFrame(metrics).T

        # ---------------- Regret compute with user settings ----------------
        Y_true = y_test
        Y_pred = best_pred
        m = Y_true.shape[1]

        # Reference point for HV
        reference_point = None
        if hv_ref_mode.startswith("Auto"):
            margin = 10.0 if hv_margin_pct is None else float(hv_margin_pct)
            reference_point = _auto_reference_point(Y_true, Y_pred, margin_pct=margin)
        else:
            try:
                ref_vals = _parse_csv_floats(hv_ref_custom or "")
                if len(ref_vals) != m:
                    raise ValueError
                reference_point = np.asarray(ref_vals, dtype=float)
            except Exception:
                st.warning("Invalid custom reference point; falling back to auto 10%.")
                reference_point = _auto_reference_point(Y_true, Y_pred, margin_pct=10.0)

        # Weights for scalarized regret
        if m == 1:
            W = 1.0
        else:
            if weights_mode == "Uniform":
                W = np.ones((1, m), dtype=float) / m
            elif weights_mode == "Custom":
                W = _parse_weight_matrix(weights_custom or "", m)
                if W is None:
                    st.warning("Invalid custom weights; falling back to uniform.")
                    W = np.ones((1, m), dtype=float) / m
            else:  # Auto (Dirichlet k=8)
                rng = np.random.default_rng(7)
                W = rng.dirichlet(alpha=np.ones(m), size=8)

        r_hv = hypervolume_regret(Y_true, Y_pred, reference_point=reference_point)
        r_sc = scalarized_regret(Y_true, Y_pred, weights=W, reduce="mean")
        r_eps = epsilon_regret(Y_true, Y_pred)

        # Parse optional bounds
        bhv = _parse_optional_float(bounds_hv)
        bsc = _parse_optional_float(bounds_scalar)
        beps = _parse_optional_float(bounds_eps)
        # -------------------------------------------------------------------

        # SHAP plot
        shap_img = safe_plot_b64(
            plot_multioutput_shap_bar_subplots,
            model,
            X_train,
            feature_names=used_feature_names,
            output_names=output_cols,
        )

        # HTML
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
            best_model=model,
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

        # Serialize model bytes
        model_bytes = _serialize_model(model)

        # Stash artifacts + settings
        st.session_state.artifacts.update(
            {
                "last_run_id": run_id,
                "model_bytes": model_bytes,
                "html_report": html,
                "metrics_df": metrics_df,
                "best_combo": best_combo,
                "y_test": y_test,
                "best_pred": best_pred,
                "best_std": best_std,
                "used_feature_names": used_feature_names,
                "output_cols": list(output_cols),
                "regret": {
                    "hv": r_hv,
                    "scalar": float(r_sc),
                    "eps": r_eps,
                    "bounds": {"hv": bhv, "scalar": bsc, "eps": beps},
                },
                "settings": {
                    "hv_ref_mode": hv_ref_mode,
                    "hv_margin_pct": hv_margin_pct,
                    "reference_point": reference_point.tolist()
                    if reference_point is not None
                    else None,
                    "weights_mode": weights_mode,
                    "weights_shape": None
                    if isinstance(W, float)
                    else list(np.asarray(W).shape),
                },
            }
        )

    # ---------- RESULTS + DOWNLOADS (shown whenever artifacts exist) ----------
    arts = st.session_state.artifacts
    if arts["model_bytes"] is not None and arts["html_report"] is not None:
        st.write("### Selected Surrogates")
        st.json(arts["best_combo"])

        st.write("### Metrics Table")
        st.dataframe(arts["metrics_df"])

        st.subheader("Multi-Objective Regret")

        def _status_line(label: str, value, bound):
            # Gracefully handle missing or None values/bounds
            if value is None:
                st.write(f"{label}: (not computed)")
                return
            try:
                v = float(value)
            except Exception:
                st.write(f"{label}: (not numeric)")
                return
            if bound is None:
                st.write(f"{label}: {v:.4g}")
            else:
                try:
                    b = float(bound)
                    ok = v <= b
                    icon = "✅" if ok else "❌"
                    st.write(f"{icon} {label}: {v:.4g} (bound ≤ {b:g})")
                except Exception:
                    st.write(f"{label}: {v:.4g} (bound not numeric)")

        # Backward-compatible access
        regret = arts.get("regret") or {}
        bounds = regret.get("bounds") or {}

        _status_line("Hypervolume regret", regret.get("hv"), bounds.get("hv"))
        _status_line("Scalarized regret (mean)", regret.get("scalar"), bounds.get("scalar"))
        _status_line("Epsilon regret", regret.get("eps"), bounds.get("eps"))

        # Settings summary (collapsed)
        with st.expander("Regret settings summary", expanded=False):
            st.json(arts.get("settings") or {})

        # Download buttons OUTSIDE the submit block so they persist across reruns
        st.download_button(
            "Download HTML Report",
            data=arts["html_report"],
            file_name="model_report_auto.html",
            mime="text/html",
            use_container_width=True,
            key="download_html_report",
        )
        st.download_button(
            "Download Trained Model (.joblib)",
            data=arts["model_bytes"],
            file_name="multioutreg_best_model.joblib",
            mime="application/octet-stream",
            use_container_width=True,
            key="download_model",
        )
