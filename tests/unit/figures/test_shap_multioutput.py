# Copyright (c) 2025 takotime808

import base64
import pytest
import matplotlib
matplotlib.use('Agg')  # Prevents GUI popup in CI/test env
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.figures.shap_multioutput import (
    plot_multioutput_shap_bar_subplots,
    generate_shap_plot,
)

@pytest.fixture(scope="module")
def fitted_model_and_data():
    X, Y = make_regression(n_samples=30, n_features=4, n_targets=3, noise=0.1, random_state=42)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=0))
    model.fit(X, Y)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    output_names = [f"t{i}" for i in range(Y.shape[1])]
    return model, X, feature_names, output_names


def test_returns_figure(fitted_model_and_data):
    model, X, feature_names, output_names = fitted_model_and_data
    fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_none_feature_and_output_names(fitted_model_and_data):
    model, X, _, _ = fitted_model_and_data
    fig = plot_multioutput_shap_bar_subplots(model, X)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_custom_max_cols(fitted_model_and_data):
    model, X, feature_names, output_names = fitted_model_and_data
    fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names, max_cols=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_savefig_creates_file(fitted_model_and_data, tmp_path):
    model, X, feature_names, output_names = fitted_model_and_data
    fpath = tmp_path / "shapbar.png"
    fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names, savefig=str(fpath))
    assert fpath.exists()
    assert isinstance(fig, plt.Figure)


def test_various_output_and_feature_counts():
    combos = [(1, 2), (4, 3), (5, 6)]
    for n_targets, n_features in combos:
        X, Y = make_regression(n_samples=20, n_features=n_features, n_targets=n_targets, noise=0.1, random_state=7)
        if n_targets == 1:
            Y = Y.reshape(-1, 1)
        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=5, random_state=11))
        model.fit(X, Y)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        output_names = [f"out_{i}" for i in range(n_targets)]
        fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_handles_large_max_cols(fitted_model_and_data):
    model, X, feature_names, output_names = fitted_model_and_data
    fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names, max_cols=50)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_value_labels_present(fitted_model_and_data):
    model, X, feature_names, output_names = fitted_model_and_data
    fig = plot_multioutput_shap_bar_subplots(model, X, feature_names, output_names)
    # Check that each axis has text labels equal to n_features
    axes = fig.get_axes()
    for ax in axes:
        texts = [t for t in ax.texts if t.get_text().replace('.', '', 1).replace('-', '', 1).isdigit()]
        assert len(texts) == len(feature_names)
    plt.close(fig)


@pytest.fixture
def multioutput_model_and_data():
    X, Y = make_regression(n_samples=100, n_features=5, n_targets=2, noise=0.1, random_state=42)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=10, random_state=42))
    model.fit(X, Y)
    output_names = [f"Target {i}" for i in range(Y.shape[1])]
    return model, X, output_names


def test_generate_shap_plot_output_keys(multioutput_model_and_data):
    model, X, output_names = multioutput_model_and_data
    plots = generate_shap_plot(model, X, output_names)
    assert isinstance(plots, dict)
    assert set(plots.keys()) == set(output_names)


def test_generate_shap_plot_values_are_base64(multioutput_model_and_data):
    model, X, output_names = multioutput_model_and_data
    plots = generate_shap_plot(model, X, output_names)
    for val in plots.values():
        assert isinstance(val, str)
        try:
            decoded = base64.b64decode(val)
            assert decoded.startswith(b'\x89PNG') or b"<svg" in decoded  # check for PNG or SVG
        except Exception as e:
            pytest.fail(f"Value is not valid base64 image: {e}")


def test_generate_shap_plot_fallback(monkeypatch, multioutput_model_and_data):
    model, X, output_names = multioutput_model_and_data

    # Break predict to force the fallback
    def broken_predict(X):
        raise ValueError("SHAP not supported")

    model.estimators_[0].predict = broken_predict

    plots = generate_shap_plot(model, X, output_names)
    assert isinstance(plots[output_names[0]], str)
    decoded = base64.b64decode(plots[output_names[0]])
    assert b"SHAP not supported" in decoded or decoded.startswith(b'\x89PNG')
