# Copyright (c) 2025 takotime808

import importlib
import sys

import pytest
import numpy as np
from unittest.mock import patch

from multioutreg.figures.umap_plot_classify import generate_umap_plot
from multioutreg.sampling.detect import SamplingDetection


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_with_input(mock_plot_to_b64, real_umap):
    """The one test that runs the real umap package (catches API breaks)."""
    assert sys.modules["umap"] is real_umap  # what the function imports
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


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_passes_umap_options(mock_plot_to_b64):
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64img")
    with patch("umap.UMAP") as mock_umap:
        mock_umap.return_value.fit_transform.side_effect = lambda X: X[:, :2]
        generate_umap_plot(np.random.rand(10, 3), n_neighbors=30, min_dist=0.5, labels=np.arange(10))
    kwargs = mock_umap.call_args.kwargs
    assert kwargs["n_neighbors"] == 9  # capped at n - 1
    assert kwargs["min_dist"] == 0.5


@patch("multioutreg.figures.umap_plot_classify.plot_to_b64")
def test_generate_umap_plot_falls_back_to_pca(mock_plot_to_b64):
    titles = []

    def run(fn):
        fn()
        import matplotlib.pyplot as plt
        titles.append(plt.gca().get_title())
        return "base64img"

    mock_plot_to_b64.side_effect = run
    with patch("umap.UMAP", side_effect=RuntimeError("boom")):
        with pytest.warns(UserWarning, match="PCA"):
            b64img, explanation = generate_umap_plot(np.random.rand(30, 3))
    assert b64img == "base64img"
    assert "->" in explanation
    assert titles[0].startswith("PCA")


def test_generate_umap_plot_without_umap_installed(monkeypatch):
    titles = []

    def run(fn):
        fn()
        import matplotlib.pyplot as plt
        titles.append(plt.gca().get_title())
        return "base64img"

    monkeypatch.setitem(sys.modules, "umap", None)  # ``import umap`` now raises ImportError
    from multioutreg.figures import umap_plot_classify

    module = importlib.reload(umap_plot_classify)  # the module itself imports without umap
    with patch.object(module, "plot_to_b64", side_effect=run):
        with pytest.warns(UserWarning, match="PCA"):
            b64img, explanation = module.generate_umap_plot(np.random.rand(30, 3))
    assert b64img == "base64img"
    assert "->" in explanation
    assert titles[0].startswith("PCA")
