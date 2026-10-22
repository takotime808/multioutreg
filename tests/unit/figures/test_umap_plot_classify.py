# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from unittest.mock import patch

from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.sampling.detect import SamplingDetection


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_with_input(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64umap")
    X = np.random.default_rng(0).random((100, 5))
    b64img, explanation = generate_umap_plot(X)
    assert b64img == "base64umap"
    assert isinstance(explanation, str)
    assert "->" in explanation
    mock_plot_to_b64.assert_called_once()


@pytest.mark.parametrize("X", [None, np.empty((0, 5))])
def test_generate_umap_plot_rejects_missing_data(X):
    # It used to plot made-up random data instead.
    with pytest.raises(ValueError):
        generate_umap_plot(X)


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
@patch("multioutreg.figures.umap_plot_classify.detect_sampling_method")
@pytest.mark.parametrize("method", ["Grid", "Random", "Sobol", "LHS", "Uncertain"])
def test_generate_umap_plot_uses_detector(mock_detect, mock_plot_to_b64, method):
    mock_detect.return_value = SamplingDetection(method=method, explanation=f"stats -> {method}")
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64img")

    _, explanation = generate_umap_plot(np.random.rand(30, 3))
    assert explanation == f"stats -> {method}"


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_too_few_points(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64img")
    _, explanation = generate_umap_plot(np.random.rand(3, 2))
    assert explanation.endswith("-> Uncertain")
