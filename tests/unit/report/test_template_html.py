# Copyright (c) 2025 takotime808

import pytest
from jinja2 import Environment, FileSystemLoader
import tempfile
import os

@pytest.fixture
def template_env():
    # Point to the actual directory containing template.html
    import multioutreg.report
    template_dir = os.path.dirname(os.path.abspath(multioutreg.report.__file__))
    return Environment(loader=FileSystemLoader(template_dir))

@pytest.fixture
def minimal_context():
    # All required fields for the template to render
    return {
        "project_title": "Test Project",
        "model_type": "DemoModel",
        "fidelity_levels": ["Low", "High"],
        "output_names": ["A", "B"],
        "description": "Test Description",
        "metrics": {
            "A": {"r2": 0.99, "rmse": 0.1, "mae": 0.09, "max_error": 0.3},
            "B": {"r2": 0.98, "rmse": 0.2, "mae": 0.15, "max_error": 0.4},
        },
        "uncertainty_metrics": {
            "NLL": 1.1, "Sharpness": 0.2, "Miscoverage": 0.05, "Calibration Error": 0.03
        },
        "uncertainty_plots": [
            {"img_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=", "title": "Calib", "caption": "Caption"}
        ],
        "prediction_plots": {
            "A": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
            "B": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        },
        "shap_plots": {
            "A": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
            "B": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        },
        "pdp_plots": {
            "A": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
            "B": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
        },
        "sampling_umap_plot": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
        "sampling_method_explanation": "Test method",
        "sampling_other_plots": [],
        "other_plots": [
            {"img_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=", "title": "ErrHist", "caption": "Error Histogram"}
        ],
        "n_train": 50,
        "n_test": 10,
        "cross_validation": "3-fold",
        "seed": 123,
        "notes": "Some notes\nWith newline."
    }

def test_template_renders(template_env, minimal_context):
    template = template_env.get_template("template.html")
    html = template.render(**minimal_context)
    # Basic checks for expected content
    assert "<title>Test Project</title>" in html
    assert "DemoModel" in html
    assert "Low, High" in html
    assert "A" in html and "B" in html
    assert "Test Description" in html
    assert "Regression Metrics" in html
    assert "Uncertainty Metrics" in html
    assert "Predictions vs. True Values" in html
    assert "SHAP Explanations" in html
    assert "Partial Dependence Plots" in html
    assert "Sampling Technique Inference" in html
    assert "Error Histogram" in html
    assert "Some notes<br>With newline." in html or "Some notes\nWith newline." in html

def test_template_handles_missing_fields(template_env, minimal_context):
    # Remove some optional fields
    context = minimal_context.copy()
    context.pop("sampling_other_plots")
    context.pop("other_plots")
    template = template_env.get_template("template.html")
    html = template.render(**context)
    # Still renders main sections
    assert "Sampling Technique Inference" in html
    assert "Appendix: Methodology & Notes" in html

def test_template_image_tags_exist(template_env, minimal_context):
    template = template_env.get_template("template.html")
    html = template.render(**minimal_context)
    # Check that image tags for all sections appear
    assert html.count("<img src=\"data:image/png;base64,") >= 6

def test_template_rounding_and_loops(template_env, minimal_context):
    template = template_env.get_template("template.html")
    html = template.render(**minimal_context)
    # Metrics rounding: "0.99", "0.1", etc. should appear
    assert "0.99" in html
    assert "0.1" in html or "0.10" in html
