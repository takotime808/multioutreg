# Copyright (c) 2025 takotime808

import pytest
from multioutreg.report.template import template

@pytest.fixture
def minimal_context():
    # All required fields for the template to render
    return {
        "project_title": "Test Py Template",
        "model_type": "TestModel",
        "fidelity_levels": ["L", "H"],
        "output_names": ["A", "B"],
        "description": "Short description.",
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
        "notes": "A note\nAnother note"
    }

def test_template_renders(minimal_context):
    html = template.render(**minimal_context)
    # Basic content
    assert "<title>Test Py Template</title>" in html
    assert "TestModel" in html
    assert "L, H" in html
    assert "A" in html and "B" in html
    assert "Short description." in html
    assert "Regression Metrics" in html
    assert "Uncertainty Metrics" in html
    assert "Predictions vs. True Values" in html
    assert "SHAP Explanations" in html
    assert "Partial Dependence Plots" in html
    assert "Sampling Technique Inference" in html
    assert "Error Histogram" in html
    assert "A note<br>Another note" in html or "A note\nAnother note" in html

def test_template_image_tags_exist(minimal_context):
    html = template.render(**minimal_context)
    # Should have an <img src= for each plot
    assert html.count("<img src=\"data:image/png;base64,") >= 7  # 2 outputs * 3 types + UMAP + 1 unc. plot + 1 other

def test_template_works_with_empty_optional_fields(minimal_context):
    context = minimal_context.copy()
    context["sampling_other_plots"] = []
    context["other_plots"] = []
    html = template.render(**context)
    assert "Sampling Technique Inference" in html
    assert "Additional Diagnostics" in html
    # Should still render without error

def test_template_metrics_and_rounding(minimal_context):
    html = template.render(**minimal_context)
    # Rounded metric values
    assert "0.99" in html
    assert "0.1" in html or "0.10" in html
