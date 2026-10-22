# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt
from multioutreg.sampling.detect import SamplingDetection
from multioutreg.sampling.infer_sampling import infer_sampling_and_plot_umap


@pytest.fixture
def sample_data():
    """Generates synthetic input data for testing."""
    rng = np.random.RandomState(0)
    return np.vstack([
        rng.normal(loc=0.0, scale=1.0, size=(50, 5)),
        rng.normal(loc=5.0, scale=1.0, size=(50, 5))
    ])


def test_infer_sampling_returns_expected_format(sample_data):
    """Test that function returns a tuple of (str, Figure) without explanation."""
    result = infer_sampling_and_plot_umap(sample_data)
    assert isinstance(result, tuple)
    assert len(result) == 2

    method, fig = result
    assert isinstance(method, str)
    assert isinstance(fig, plt.Figure)


def test_infer_sampling_with_explanation(sample_data):
    """Test that function returns a tuple of (str, Figure, str) when explanation is requested."""
    result = infer_sampling_and_plot_umap(sample_data, explanation_indicator=True)
    assert isinstance(result, tuple)
    assert len(result) == 3

    method, fig, explanation = result
    assert isinstance(method, str)
    assert isinstance(fig, plt.Figure)
    assert isinstance(explanation, str)
    assert method in explanation or method == "Uncertain"


def test_infer_sampling_method_string(sample_data):
    """Test that the inferred method string is non-empty and reasonably short."""
    method, _ = infer_sampling_and_plot_umap(sample_data)
    assert isinstance(method, str)
    assert 0 < len(method) < 100  # sanity check


def test_umap_plot_structure(sample_data):
    """Ensure the generated figure has a valid structure and contains scatter points."""
    _, fig = infer_sampling_and_plot_umap(sample_data)
    ax = fig.axes[0]
    assert ax.get_title().startswith("UMAP 2D Projection")
    assert ax.get_xlabel() == "UMAP-1"
    assert ax.get_ylabel() == "UMAP-2"
    assert len(ax.collections) > 0  # check scatter plot present


@patch("multioutreg.sampling.infer_sampling.detect_sampling_method")
def test_infer_sampling_uses_detector_result(mock_detect):
    """Method and explanation come straight from the statistical detector."""
    mock_detect.return_value = SamplingDetection(method="Grid", explanation="levels -> Grid")

    X = np.random.rand(10, 5)
    method, fig, returned_explanation = infer_sampling_and_plot_umap(X, explanation_indicator=True)
    assert method == "Grid"
    assert returned_explanation == "levels -> Grid"
    assert isinstance(fig, plt.Figure)


def test_infer_sampling_two_clusters_is_not_a_design(sample_data):
    """Two Gaussian blobs are not a uniform design of any kind."""
    method, _ = infer_sampling_and_plot_umap(sample_data)
    assert method == "Uncertain"
