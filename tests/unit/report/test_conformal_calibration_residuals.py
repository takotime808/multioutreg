# Copyright (c) 2026 takotime808

import numpy as np
import pytest

import multioutreg.gui.grid_search_core as grid_core
import multioutreg.report.auto_detect_report as auto_report
from multioutreg.figures.conformal_plots import plot_conformal_vs_gaussian


@pytest.fixture(params=[grid_core, auto_report], ids=["grid_search_core", "auto_detect_report"])
def report_module(request, monkeypatch):
    mod = request.param
    calls = []

    def fake_safe_plot_b64(fn, *args, **kwargs):
        calls.append((fn, args))
        return ""

    monkeypatch.setattr(mod, "safe_plot_b64", fake_safe_plot_b64)
    per_output = lambda *a, **k: {"a": "", "b": ""}
    monkeypatch.setattr(mod, "generate_shap_plot", per_output, raising=False)
    monkeypatch.setattr(mod, "generate_pdp_plot", per_output)
    monkeypatch.setattr(mod, "generate_umap_plot", lambda *a, **k: ("", ""))
    monkeypatch.setattr(mod, "generate_error_histogram", lambda *a, **k: [])
    return mod, calls


def _render(mod, **kwargs):
    rng = np.random.default_rng(0)
    y_test = rng.normal(size=(20, 2))
    pred = y_test + 0.1 * rng.normal(size=(20, 2))
    return mod.generate_html_report(
        model_type="m", fidelity_levels=[], output_names=["a", "b"],
        description="",
        metrics={n: {"r2": 1.0, "rmse": 0.1, "mae": 0.1} for n in "ab"}, uncertainty_metrics={},
        y_test=y_test, best_pred=pred, best_std=np.full((20, 2), 0.1),
        best_model=None, X_train=rng.normal(size=(30, 2)), n_train=30, n_test=20,
        cross_validation="", seed=0, notes="",
        conformal_intervals=(pred - 0.2, pred + 0.2), conformal_alpha=0.1,
        **kwargs,
    )


def _vs_gaussian_calls(calls):
    return [args for fn, args in calls if fn is plot_conformal_vs_gaussian]


def test_conformal_vs_gaussian_uses_calibration_residuals(report_module):
    # MF11: the plot used |y_test - best_pred| as calibration residuals, so the
    # conformal curve matched the diagonal by construction.
    mod, calls = report_module
    cal = np.full((15, 2), 7.0)
    _render(mod, calibration_residuals=cal)
    (args,) = _vs_gaussian_calls(calls)
    assert args[3] is cal


def test_conformal_vs_gaussian_skipped_without_calibration_residuals(report_module):
    mod, calls = report_module
    _render(mod)
    assert _vs_gaussian_calls(calls) == []
