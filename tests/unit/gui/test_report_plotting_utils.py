# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
# from sklearn.decomposition import PCA
from multioutreg.gui import report_plotting_utils as utils


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = np.random.rand(100, 2)
    output_names = ["Target 1", "Target 2"]
    return X, y, output_names


def test_generate_prediction_plot(sample_regression_data):
    X, y, output_names = sample_regression_data
    y_pred = y + 0.1 * np.random.randn(*y.shape)
    y_std = 0.05 * np.ones_like(y)
    plots = utils.generate_prediction_plot(y, y_pred, y_std, output_names)
    assert isinstance(plots, dict)
    assert set(plots.keys()) == set(output_names)
    for b64 in plots.values():
        assert isinstance(b64, str)
        assert len(b64) > 100


def test_generate_pdp_plot(sample_regression_data):
    X, y, output_names = sample_regression_data
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    model = MultiOutputRegressor(LinearRegression()).fit(X, y)
    plots = utils.generate_pdp_plot(model, X, output_names, feature_names)
    assert isinstance(plots, dict)
    assert set(plots.keys()) == set(output_names)
    for b64 in plots.values():
        assert isinstance(b64, str)
        assert len(b64) > 100


def test_generate_uncertainty_plots():
    plots = utils.generate_uncertainty_plots()
    assert isinstance(plots, list)
    assert len(plots) == 1
    assert "img_b64" in plots[0]
    assert isinstance(plots[0]["img_b64"], str)
    assert len(plots[0]["img_b64"]) > 100


def test_generate_umap_plot():
    X = np.random.normal(size=(100, 5))
    b64_img, caption = utils.generate_umap_plot(X)
    assert isinstance(b64_img, str)
    assert len(b64_img) > 100
    assert isinstance(caption, str)
    assert any(method in caption.lower() for method in ["grid", "lhs", "sobol", "random", "uncertain"])


def test_generate_error_histogram(sample_regression_data):
    X, y, output_names = sample_regression_data
    y_pred = y + np.random.normal(0, 0.1, size=y.shape)
    plots = utils.generate_error_histogram(y, y_pred, output_names)
    assert isinstance(plots, list)
    assert len(plots) == len(output_names)
    for plot in plots:
        assert "img_b64" in plot
        assert isinstance(plot["img_b64"], str)
        assert len(plot["img_b64"]) > 100


def test_generate_umap_plot_with_none():
    b64_img, caption = utils.generate_umap_plot(None)
    assert isinstance(b64_img, str)
    assert isinstance(caption, str)


def test_plot_to_b64_direct():
    def simple_plot():
        plt.plot([0, 1], [0, 1])
    b64_img = utils.plot_to_b64(simple_plot)
    assert isinstance(b64_img, str)
    assert len(b64_img) > 100


class DummyModel:
    pass

def test_generate_pdp_plot_fallback(sample_regression_data):
    X, _, output_names = sample_regression_data
    feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    dummy_model = type("Dummy", (), {"estimators_": [DummyModel(), DummyModel()]})
    plots = utils.generate_pdp_plot(dummy_model, X, output_names, feature_names)
    for b64 in plots.values():
        assert isinstance(b64, str)
        assert len(b64) > 100  # should still return a fallback image
