# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from unittest.mock import patch
from scipy.stats import qmc

from multioutreg.sampling.detect import SamplingDetection
from multioutreg.sampling.pca_sampling_detector import (
    embedding_diagnostics,
    generate_sampling_plot_and_metrics,
    label_colour_values,
    pca_embedding,
)


@pytest.mark.parametrize("label, sampler", [
    ("Grid", lambda: np.stack(np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20)), -1).reshape(-1, 2)),
    ("Sobol", lambda: qmc.Sobol(d=2, scramble=False).random(256)),
    ("Uncertain", lambda: np.ones((50, 2))),
])
def test_deterministic_designs_detected(label, sampler):
    _, metrics = generate_sampling_plot_and_metrics(sampler(), random_state=42)
    assert metrics["method"] == label, metrics["explanation"]


@pytest.mark.parametrize("label, sampler", [
    ("Random", lambda: np.random.default_rng(0).random((200, 2))),
    ("LHS", lambda: qmc.LatinHypercube(d=2, seed=0).random(200)),
])
def test_stochastic_designs_detected(label, sampler):
    _, metrics = generate_sampling_plot_and_metrics(sampler(), random_state=42)
    assert metrics["method"] == label, metrics["explanation"]
    assert metrics["p_values"][label] >= 0.05


@patch("multioutreg.sampling.pca_sampling_detector.detect_sampling_method")
def test_method_comes_from_detector(mock_detect):
    mock_detect.return_value = SamplingDetection(method="LHS", explanation="stats -> LHS")
    _, metrics = generate_sampling_plot_and_metrics(np.random.rand(30, 3))
    assert metrics["method"] == "LHS"
    assert metrics["explanation"] == "stats -> LHS"


def test_output_structure():
    fig, metrics = generate_sampling_plot_and_metrics(np.random.rand(100, 3))
    assert hasattr(fig, "gca"), "Output is not a matplotlib figure"
    for key in ["method", "explanation", "projection", "explained_variance",
                "p_values", "std", "silhouette", "entropy"]:
        assert key in metrics
    assert metrics["projection"] == "PCA"
    assert len(metrics["explained_variance"]) == 2


def test_too_few_points_is_uncertain():
    _, metrics = generate_sampling_plot_and_metrics(np.random.rand(3, 2))
    assert metrics["method"] == "Uncertain"
    assert metrics["explanation"].endswith("-> Uncertain")


@pytest.mark.parametrize("X", [None, np.empty((0, 2))])
def test_rejects_missing_data(X):
    with pytest.raises(ValueError):
        generate_sampling_plot_and_metrics(X)


def test_single_input_and_labels():
    X = np.random.rand(40)
    labels = np.array(["a", "b"] * 20)
    fig, metrics = generate_sampling_plot_and_metrics(X, labels=labels)
    assert len(metrics["explained_variance"]) == 1
    assert len(fig.axes) == 2  # scatter + colorbar


def test_pca_embedding_shapes():
    X_emb, evr = pca_embedding(np.random.rand(20, 4))
    assert X_emb.shape == (20, 2)
    assert np.all(np.diff(evr) <= 0)
    X_emb, evr = pca_embedding(np.ones((10, 3)))
    assert not X_emb.any() and not evr.any()


def test_embedding_diagnostics_constant():
    diag = embedding_diagnostics(np.zeros((10, 2)))
    assert diag["std"] == 0.0
    assert diag["silhouette"] == 0.0


def test_label_colour_values():
    np.testing.assert_array_equal(label_colour_values(["b", "a", "b"]), [1, 0, 1])
    np.testing.assert_array_equal(label_colour_values([3, 1]), [3.0, 1.0])
