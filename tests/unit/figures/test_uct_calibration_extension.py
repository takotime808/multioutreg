# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless/CI test
import matplotlib.pyplot as plt
import pytest

from multioutreg.figures import uct_calibration_extension as uce


def test_filter_subset_reduces_size():
    arrs = [np.arange(10), np.arange(10)*2, np.arange(10)*3]
    sub = uce.filter_subset(arrs, 5)
    assert all(len(a) == 5 for a in sub)
    assert all(isinstance(a, np.ndarray) for a in sub)


def test_get_proportion_lists_vectorized_shape_and_range():
    np.random.seed(1)
    y_pred = np.random.normal(size=30)
    y_std = np.abs(np.random.normal(1, 0.1, size=30))
    y_true = y_pred + np.random.normal(0, 1, size=30)
    exp, obs = uce.get_proportion_lists_vectorized(y_pred, y_std, y_true)
    assert exp.shape == obs.shape
    assert np.all(exp >= 0) and np.all(exp <= 1)
    assert np.all(obs >= 0) and np.all(obs <= 1)
    assert len(exp) == 100


def test_miscalibration_area_from_proportions_behavior():
    # Perfect calibration: area should be 0
    exp = np.linspace(0, 1, 100)
    obs = np.linspace(0, 1, 100)
    area = uce.miscalibration_area_from_proportions(exp, obs)
    assert np.isclose(area, 0)
    # Max miscalibration: diagonal vs zeros
    obs2 = np.zeros(100)
    area2 = uce.miscalibration_area_from_proportions(exp, obs2)
    assert area2 > 0.45


def test_plot_calibration_basic():
    np.random.seed(0)
    y_pred = np.random.normal(0, 1, 50)
    y_std = np.abs(np.random.normal(1, 0.1, 50))
    y_true = y_pred + np.random.normal(0, 1, 50)
    ax = uce.plot_calibration(y_pred, y_std, y_true)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_calibration_with_ax_and_label():
    np.random.seed(2)
    y_pred = np.random.normal(0, 1, 40)
    y_std = np.abs(np.random.normal(0.5, 0.1, 40))
    y_true = y_pred + np.random.normal(0, 1, 40)
    fig, ax = plt.subplots()
    ax_out = uce.plot_calibration(y_pred, y_std, y_true, ax=ax, curve_label="TestLabel")
    assert ax_out is ax
    plt.close()


def test_plot_calibration_with_n_subset():
    np.random.seed(3)
    y_pred = np.random.normal(size=100)
    y_std = np.abs(np.random.normal(1, 0.2, size=100))
    y_true = y_pred + np.random.normal(0, 1, 100)
    ax = uce.plot_calibration(y_pred, y_std, y_true, n_subset=20)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_calibration_with_custom_props():
    exp = np.linspace(0, 1, 20)
    obs = np.linspace(0, 1, 20) ** 2
    y_pred = np.random.normal(0, 1, 20)
    y_std = np.abs(np.random.normal(1, 0.1, 20))
    y_true = y_pred + np.random.normal(0, 1, 20)
    ax = uce.plot_calibration(y_pred, y_std, y_true, exp_props=exp, obs_props=obs)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close()


def test_plot_calibration_raises_on_shape_mismatch():
    exp = np.linspace(0, 1, 10)
    obs = np.linspace(0, 1, 11)
    y_pred = np.random.normal(0, 1, 11)
    y_std = np.abs(np.random.normal(1, 0.1, 11))
    y_true = y_pred + np.random.normal(0, 1, 11)
    with pytest.raises(RuntimeError):
        uce.plot_calibration(y_pred, y_std, y_true, exp_props=exp, obs_props=obs)


def test_plot_calibration_vectorized_only():
    y_pred = np.random.normal(0, 1, 20)
    y_std = np.abs(np.random.normal(1, 0.1, 20))
    y_true = y_pred + np.random.normal(0, 1, 20)
    with pytest.raises(NotImplementedError):
        uce.plot_calibration(y_pred, y_std, y_true, vectorized=False)
