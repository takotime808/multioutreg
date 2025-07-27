# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import base64
from multioutreg.figures.error_histograms import generate_error_histogram

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    y_true = np.random.rand(100, 3)
    y_pred = y_true + np.random.normal(0, 0.1, size=y_true.shape)
    output_names = ["output_1", "output_2", "output_3"]
    return y_true, y_pred, output_names

def test_generate_error_histogram_structure(synthetic_data):
    y_true, y_pred, output_names = synthetic_data
    result = generate_error_histogram(y_true, y_pred, output_names)

    assert isinstance(result, list), "Result should be a list"
    assert len(result) == len(output_names), "Result should contain one plot per output name"

    for entry, name in zip(result, output_names):
        assert isinstance(entry, dict), "Each entry should be a dictionary"
        assert "img_b64" in entry, "'img_b64' key missing"
        assert "title" in entry, "'title' key missing"
        assert "caption" in entry, "'caption' key missing"
        assert name in entry["caption"], f"Output name '{name}' should appear in caption"
        assert entry["img_b64"].startswith("iVBOR") or entry["img_b64"].startswith("data:image"), \
            "Image base64 string does not start with expected PNG/JPEG prefix"

def test_generate_error_histogram_content_not_empty(synthetic_data):
    y_true, y_pred, output_names = synthetic_data
    result = generate_error_histogram(y_true, y_pred, output_names)

    for entry in result:
        b64_str = entry["img_b64"]
        try:
            # Try decoding the base64 string
            _ = base64.b64decode(b64_str)
        except Exception as e:
            pytest.fail(f"Base64 decoding failed: {e}")
