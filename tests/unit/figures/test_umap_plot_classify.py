# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from unittest.mock import patch

from multioutreg.figures.umap_plot_classify import generate_umap_plot


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_with_input(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64umap")
    X = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    b64img, explanation = generate_umap_plot(X)
    assert b64img == "base64umap"
    assert isinstance(explanation, str)
    assert "->" in explanation or "Pattern unclear" in explanation
    mock_plot_to_b64.assert_called_once()


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_with_none(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64umap")
    b64img, explanation = generate_umap_plot(None)
    assert b64img == "base64umap"
    assert isinstance(explanation, str)
    mock_plot_to_b64.assert_called_once()


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_with_empty(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64umap")
    b64img, explanation = generate_umap_plot(np.empty((0, 5)))
    assert b64img == "base64umap"
    assert isinstance(explanation, str)
    mock_plot_to_b64.assert_called_once()


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
@patch("multioutreg.figures.umap_plot_classify.entropy")
@patch("multioutreg.figures.umap_plot_classify.silhouette_score")
@patch("multioutreg.figures.umap_plot_classify.np.std")
@pytest.mark.parametrize("mock_std_val, mock_sil_val, mock_ent_val, expected_keyword", [
    (0.01, 0.9, 1.0, "Grid"),
    (0.3, 0.2, 2.5, "Random"),
    (0.10, 0.5, 1.5, "Sobol"),
    (0.10, 0.5, 2.5, "LHS"),
    (0.18, 0.4, 1.5, "Uncertain"),
])
def test_generate_umap_plot_explanation_paths(
    mock_std, mock_silhouette, mock_entropy,
    mock_plot_to_b64, mock_std_val, mock_sil_val, mock_ent_val, expected_keyword
):
    mock_std.return_value = mock_std_val
    mock_silhouette.return_value = mock_sil_val
    mock_entropy.return_value = mock_ent_val
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64img")

    X = np.random.rand(100, 5)
    _, explanation = generate_umap_plot(X)

    expected_explanation = {
        "Grid": "Low std and high silhouette -> Grid",
        "Random": "High std and low silhouette -> Random",
        "Sobol": "Moderate spread, low entropy -> Sobol",
        "LHS": "Moderate spread, higher entropy -> LHS",
        "Uncertain": "Pattern unclear"
    }[expected_keyword]

    assert explanation == expected_explanation
