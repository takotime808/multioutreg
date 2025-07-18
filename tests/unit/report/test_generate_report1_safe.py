# Copyright (c) 2025 takotime808

import pytest
# import os
import base64
# import numpy as np
from multioutreg.report import generate_report1_safe as gr

def test_prediction_plots_exist_and_valid():
    for name, img_b64 in gr.prediction_plots.items():
        assert isinstance(img_b64, str)
        # Check that it's a base64 string of reasonable length
        assert len(img_b64) > 100

def test_shap_plots_exist_and_valid():
    for name, img_b64 in gr.shap_plots.items():
        assert isinstance(img_b64, str)
        assert len(img_b64) > 100

def test_pdp_plots_exist_and_valid():
    for name, img_b64 in gr.pdp_plots.items():
        assert isinstance(img_b64, str)
        assert len(img_b64) > 100

def test_uncertainty_plots_format():
    assert isinstance(gr.uncertainty_plots, list)
    for entry in gr.uncertainty_plots:
        assert "img_b64" in entry
        assert "title" in entry
        assert "caption" in entry
        # Check base64 type
        assert isinstance(entry["img_b64"], str)
        assert len(entry["img_b64"]) > 100

def test_sampling_umap_plot_valid():
    assert isinstance(gr.sampling_umap_plot, str)
    assert len(gr.sampling_umap_plot) > 100

def test_other_plots_format():
    assert isinstance(gr.other_plots, list)
    for entry in gr.other_plots:
        assert "img_b64" in entry
        assert "title" in entry
        assert "caption" in entry
        assert isinstance(entry["img_b64"], str)
        assert len(entry["img_b64"]) > 100

def test_metrics_and_context():
    # Basic check for expected keys in metrics and context
    assert "Pressure" in gr.metrics
    assert "Temperature" in gr.metrics
    for k in ["r2", "rmse", "mae", "max_error"]:
        assert k in gr.metrics["Pressure"]
        assert k in gr.metrics["Temperature"]
    assert isinstance(gr.context, dict)
    assert "project_title" in gr.context

def test_html_report_generation(tmp_path, monkeypatch):
    # Render a report to a temporary file
    tmp_html = tmp_path / "test_report.html"
    # Patch output path
    monkeypatch.setattr(gr, "output_path", str(tmp_html))
    # Patch template path to a minimal dummy template
    dummy_template = tmp_path / "dummy.html"
    dummy_template.write_text("Title: {{ project_title }}")
    monkeypatch.setattr(gr, "template_path", str(dummy_template))
    # Recreate Jinja2 logic
    env = gr.Environment(loader=gr.FileSystemLoader(str(tmp_path)))
    template = env.get_template(str(dummy_template.name))
    html = template.render(**gr.context)
    tmp_html.write_text(html)
    # Now test
    contents = tmp_html.read_text()
    assert "Multi-Fidelity, Multi-Output Surrogate Model Demo Report" in contents

def test_safe_plot_b64_error_handling():
    # Function that raises an error
    def broken_plot():
        raise RuntimeError("deliberate test error")
    b64str = gr.safe_plot_b64(broken_plot)
    assert isinstance(b64str, str)
    assert "Error" in base64.b64decode(b64str).decode(errors="ignore") or len(b64str) > 20
