# Copyright (c) 2025 takotime808

import pytest
# import sys
# from pathlib import Path
# import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from multioutreg.figures.prediction_plots import (
    plot_predictions,
    plot_predictions_with_error_bars,
)

# ROOT = Path(__file__).resolve().parents[2]
# spec = importlib.util.spec_from_file_location(
#     "perf",
#     ROOT / "multioutreg" / "figures" / "performance_metric_figures.py",
# )
# perf = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(perf)
# plot_predictions = perf.plot_predictions


def test_plot_predictions_single_output():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    y = X @ rng.rand(3) + rng.randn(20) * 0.1
    model = LinearRegression().fit(X, y)
    ax = plot_predictions(model, X, y)
    assert ax.get_title()


def test_plot_predictions_multi_output():
    rng = np.random.RandomState(1)
    X = rng.rand(20, 3)
    Y = X @ rng.rand(3, 2) + rng.randn(20, 2) * 0.1
    model = MultiOutputRegressor(LinearRegression()).fit(X, Y)
    ax = plot_predictions(model, X, Y)
    assert ax.get_title()


def make_single_output_data(n=30):
    y_true = np.linspace(0, 1, n)
    preds = y_true + 0.1 * np.random.randn(n)
    std = 0.2 * np.ones(n)
    return y_true, preds, std

def make_multi_output_data(n=30, n_targets=2):
    y_true = np.linspace(0, 1, n)[:, None] + np.arange(n_targets)[None, :]
    preds = y_true + 0.1 * np.random.randn(n, n_targets)
    std = 0.2 * np.ones((n, n_targets))
    return y_true, preds, std

def test_single_output_default():
    y_true, preds, std = make_single_output_data()
    plot_predictions_with_error_bars(y_true, preds, std)
    plt.close('all')

def test_single_output_custom_names():
    y_true, preds, std = make_single_output_data()
    plot_predictions_with_error_bars(y_true, preds, std, output_names=['custom_name'])
    plt.close('all')

def test_single_output_list_input():
    y_true, preds, std = make_single_output_data()
    plot_predictions_with_error_bars(list(y_true), list(preds), list(std))
    plt.close('all')

def test_multi_output_default():
    y_true, preds, std = make_multi_output_data()
    plot_predictions_with_error_bars(y_true, preds, std)
    plt.close('all')

def test_multi_output_custom_names():
    y_true, preds, std = make_multi_output_data(n_targets=3)
    plot_predictions_with_error_bars(y_true, preds, std, output_names=['a', 'b', 'c'])
    plt.close('all')

def test_multi_output_list_input():
    y_true, preds, std = make_multi_output_data(n_targets=2)
    plot_predictions_with_error_bars(y_true.tolist(), preds.tolist(), std.tolist())
    plt.close('all')

def test_multi_output_more_targets_than_cols():
    y_true, preds, std = make_multi_output_data(n_targets=5)
    plot_predictions_with_error_bars(y_true, preds, std, n_cols=2)
    plt.close('all')

def test_single_output_as_2d():
    y_true, preds, std = make_single_output_data()
    plot_predictions_with_error_bars(y_true[:, None], preds[:, None], std[:, None])
    plt.close('all')

def test_multi_output_hide_unused():
    y_true, preds, std = make_multi_output_data(n_targets=4)
    # Use 3 cols, will create 2x3 grid, so 2 unused subplots
    plot_predictions_with_error_bars(y_true, preds, std, n_cols=3)
    plt.close('all')
