# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # for headless test environments

from multioutreg.figures.coverage_plots import plot_coverage

def make_single_output_data(n=50):
    rng = np.random.RandomState(42)
    y_true = rng.randn(n)
    y_pred = y_true + 0.1 * rng.randn(n)
    y_std = 0.5 + 0.1 * np.abs(rng.randn(n))
    return y_true, y_pred, y_std

def make_multi_output_data(n=50, n_targets=2):
    rng = np.random.RandomState(42)
    y_true = rng.randn(n, n_targets)
    y_pred = y_true + 0.1 * rng.randn(n, n_targets)
    y_std = 0.5 + 0.1 * np.abs(rng.randn(n, n_targets))
    return y_true, y_pred, y_std

def test_plot_coverage_single_default():
    y_true, y_pred, y_std = make_single_output_data()
    plot_coverage(y_true, y_pred, y_std)

def test_plot_coverage_single_custom_intervals():
    y_true, y_pred, y_std = make_single_output_data()
    plot_coverage(y_true, y_pred, y_std, intervals=[0.5, 0.8, 0.95])

def test_plot_coverage_single_custom_names():
    y_true, y_pred, y_std = make_single_output_data()
    plot_coverage(y_true, y_pred, y_std, output_names=["target_a"])

def test_plot_coverage_multi_default():
    y_true, y_pred, y_std = make_multi_output_data()
    plot_coverage(y_true, y_pred, y_std)

def test_plot_coverage_multi_custom_intervals():
    y_true, y_pred, y_std = make_multi_output_data()
    plot_coverage(y_true, y_pred, y_std, intervals=[0.7, 0.95])

def test_plot_coverage_multi_custom_names():
    y_true, y_pred, y_std = make_multi_output_data()
    plot_coverage(y_true, y_pred, y_std, output_names=["out1", "out2"])

def test_plot_coverage_single_one_sample():
    y_true, y_pred, y_std = make_single_output_data(n=1)
    plot_coverage(y_true, y_pred, y_std)

def test_plot_coverage_multi_one_sample():
    y_true, y_pred, y_std = make_multi_output_data(n=1)
    plot_coverage(y_true, y_pred, y_std)

def test_plot_coverage_multi_one_interval():
    y_true, y_pred, y_std = make_multi_output_data()
    plot_coverage(y_true, y_pred, y_std, intervals=[0.8])

def test_plot_coverage_single_as_2d_shape():
    y_true, y_pred, y_std = make_single_output_data()
    plot_coverage(y_true[:, None], y_pred[:, None], y_std[:, None])

def test_plot_coverage_multi_as_list():
    y_true, y_pred, y_std = make_multi_output_data()
    plot_coverage(y_true.tolist(), y_pred.tolist(), y_std.tolist())
