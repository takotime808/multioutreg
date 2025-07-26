# Copyright (c) 2025 takotime808

import pytest
import os
import base64
from multioutreg.report import generate_report1 as gr

def test_prediction_plots_exist_and_valid():
    for name, img_b64 in gr.prediction_plots.items():
        assert isinstance(img_b64, str)
        assert len(img_b64) > 100
        # Try decoding (may not be a valid PNG, but should be decodable)
        try:
            base64.b64decode(img_b64)
        except Exception:
            pytest.fail(f"Base64 string for {name} not decodable")

def test_shap_plots_exist_and_valid():
    for name, img_b64 in gr.shap_plots.items():
        assert isinstance(img_b64, str)
        assert len(img_b64) > 100
        try:
            base64.b64decode(img_b64)
        except Exception:
            pytest.fail(f"Base64 string for {name} not decodable")

def test_pdp_plots_exist_and_valid():
    for name, img_b64 in gr.pdp_plots.items():
        assert isinstance(img_b64, str)
        assert len(img_b64) > 100
        try:
            base64.b64decode(img_b64)
        except Exception:
            pytest.fail(f"Base64 string for {name} not decodable")

def test_uncertainty_plots_format():
    assert isinstance(gr.uncertainty_plots, list)
    for entry in gr.uncertainty_plots:
        assert "img_b64" in entry
        assert "title" in entry
        assert "caption" in entry
        assert isinstance(entry["img_b64"], str)
        assert len(entry["img_b64"]) > 100
        try:
            base64.b64decode(entry["img_b64"])
        except Exception:
            pytest.fail("Uncertainty plot base64 not decodable")

def test_sampling_umap_plot_valid():
    assert isinstance(gr.sampling_umap_plot, str)
    assert len(gr.sampling_umap_plot) > 100
    try:
        base64.b64decode(gr.sampling_umap_plot)
    except Exception:
        pytest.fail("Sampling UMAP base64 not decodable")

def test_other_plots_format():
    assert isinstance(gr.other_plots, list)
    for entry in gr.other_plots:
        assert "img_b64" in entry
        assert "title" in entry
        assert "caption" in entry
        assert isinstance(entry["img_b64"], str)
        assert len(entry["img_b64"]) > 100
        try:
            base64.b64decode(entry["img_b64"])
        except Exception:
            pytest.fail("Other plot base64 not decodable")

def test_metrics_and_context():
    # Basic check for expected keys in metrics and context
    assert "Pressure" in gr.metrics
    assert "Temperature" in gr.metrics
    for k in ["r2", "rmse", "mae", "max_error"]:
        assert k in gr.metrics["Pressure"]
        assert k in gr.metrics["Temperature"]
    assert isinstance(gr.context, dict)
    assert "project_title" in gr.context
    assert gr.context["metrics"] == gr.metrics

def test_html_report_generation(tmp_path, monkeypatch):
    # Patch output path and template path to use dummy template and temp file
    tmp_html = tmp_path / "test_report1.html"
    dummy_template = tmp_path / "dummy_template.html"
    dummy_template.write_text("Title: {{ project_title }}\nMetrics: {{ metrics.keys()|list }}")
    monkeypatch.setattr(gr, "output_path", str(tmp_html))
    monkeypatch.setattr(gr, "template_path", str(dummy_template))
    # Patch env to use tmp_path
    env = gr.Environment(loader=gr.FileSystemLoader(str(tmp_path)))
    template = env.get_template(dummy_template.name)
    html = template.render(**gr.context)
    tmp_html.write_text(html)
    # Now check
    contents = tmp_html.read_text()
    assert "Multi-Fidelity, Multi-Output Surrogate Model Demo Report" in contents
    assert "Pressure" in contents
    assert "Temperature" in contents
    assert "Metrics" in contents
