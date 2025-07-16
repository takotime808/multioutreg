# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Avoid GUI issues in test envs
import matplotlib.pyplot as plt
import pytest

from multioutreg.figures import uncertainty_toolbox_extension as ute


def test_plot_intervals_ordered_basic():
    y_pred = np.linspace(0, 1, 10)
    y_std = np.ones(10) * 0.2
    y_true = np.linspace(0, 1, 10) + 0.1
    ax = ute.plot_intervals_ordered(y_pred, y_std, y_true)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_intervals_ordered_with_ax():
    y_pred = np.linspace(0, 1, 8)
    y_std = np.ones(8) * 0.15
    y_true = np.linspace(0, 1, 8)
    fig, ax = plt.subplots()
    out_ax = ute.plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
    assert out_ax is ax
    plt.close()


def test_plot_intervals_ordered_with_n_subset():
    y_pred = np.linspace(0, 1, 50)
    y_std = np.ones(50) * 0.1
    y_true = np.linspace(0, 1, 50)
    ax = ute.plot_intervals_ordered(y_pred, y_std, y_true, n_subset=10)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_intervals_ordered_with_ylims():
    y_pred = np.linspace(0, 1, 12)
    y_std = np.ones(12) * 0.12
    y_true = np.linspace(0, 1, 12)
    ax = ute.plot_intervals_ordered(y_pred, y_std, y_true, ylims=(-2, 2))
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_uct_intervals_ordered_multioutput_basic():
    y_pred = np.random.randn(30, 3)
    y_std = np.abs(np.random.randn(30, 3)) * 0.1 + 0.05
    y_true = y_pred + np.random.normal(0, 0.2, size=(30, 3))
    axes = ute.plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, num_stds_confidence_bound=2)
    assert isinstance(axes, list)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in axes)
    plt.close()


def test_plot_uct_intervals_ordered_multioutput_single_output():
    y_pred = np.random.randn(20)
    y_std = np.abs(np.random.randn(20)) * 0.1 + 0.1
    y_true = y_pred + np.random.normal(0, 0.2, size=20)
    axes = ute.plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert isinstance(axes, list)
    assert isinstance(axes[0], matplotlib.axes.Axes)
    plt.close()


def test_plot_uct_intervals_ordered_multioutput_invalid_shape_raises():
    y_pred = np.random.randn(10, 2)
    y_std = np.random.randn(10, 2)
    y_true = np.random.randn(10)  # Wrong shape
    with pytest.raises(ValueError):
        ute.plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert True


def test_plot_uct_intervals_ordered_multioutput_shape_mismatch_raises():
    y_pred = np.random.randn(10, 2)
    y_std = np.random.randn(10, 2)
    y_true = np.random.randn(9, 2)  # Shape mismatch
    with pytest.raises(ValueError):
        ute.plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert True

def test_plot_uct_intervals_ordered_multioutput_with_ax_list():
    y_pred = np.random.randn(14, 2)
    y_std = np.abs(np.random.randn(14, 2)) * 0.05 + 0.07
    y_true = y_pred + np.random.normal(0, 0.1, size=(14, 2))
    fig, axes = plt.subplots(1, 2)
    out_axes = ute.plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, ax_list=axes)
    assert isinstance(out_axes, list)
    assert all(isinstance(ax, matplotlib.axes.Axes) for ax in out_axes)
    plt.close()
