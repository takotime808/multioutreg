# Copyright (c) 2025 takotime808

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multioutreg.figures.conformal_plots import (
    plot_conformal_intervals,
    plot_conformal_intervals_ordered,
    plot_conformal_coverage,
    plot_conformal_vs_gaussian,
    plot_conditional_coverage,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


@pytest.fixture
def single_output_data():
    rng = np.random.RandomState(42)
    n = 50
    y_true = rng.randn(n)
    y_pred = y_true + 0.1 * rng.randn(n)
    y_lower = y_pred - 1.5
    y_upper = y_pred + 1.5
    y_std = np.ones(n) * 0.5
    residuals = np.abs(y_true - y_pred)
    return y_true, y_pred, y_lower, y_upper, y_std, residuals


@pytest.fixture
def multi_output_data():
    rng = np.random.RandomState(42)
    n, m = 50, 3
    y_true = rng.randn(n, m)
    y_pred = y_true + 0.1 * rng.randn(n, m)
    y_lower = y_pred - 1.5
    y_upper = y_pred + 1.5
    y_std = np.ones((n, m)) * 0.5
    residuals = np.abs(y_true - y_pred)
    return y_true, y_pred, y_lower, y_upper, y_std, residuals


class TestPlotConformalIntervals:
    def test_single_output(self, single_output_data, tmp_path):
        y_true, y_pred, y_lower, y_upper, _, _ = single_output_data
        path = str(tmp_path / "test.png")
        plot_conformal_intervals(y_true, y_lower, y_upper, y_pred=y_pred, savefig=path)

    def test_multi_output(self, multi_output_data, tmp_path):
        y_true, y_pred, y_lower, y_upper, _, _ = multi_output_data
        path = str(tmp_path / "test.png")
        plot_conformal_intervals(y_true, y_lower, y_upper, y_pred=y_pred, savefig=path)

    def test_without_predictions(self, single_output_data, tmp_path):
        y_true, _, y_lower, y_upper, _, _ = single_output_data
        path = str(tmp_path / "test.png")
        plot_conformal_intervals(y_true, y_lower, y_upper, savefig=path)


class TestPlotConformalIntervalsOrdered:
    def test_single_output(self, single_output_data, tmp_path):
        y_true, y_pred, y_lower, y_upper, _, _ = single_output_data
        path = str(tmp_path / "test.png")
        plot_conformal_intervals_ordered(
            y_true, y_lower, y_upper, y_pred=y_pred, savefig=path
        )

    def test_multi_output(self, multi_output_data, tmp_path):
        y_true, y_pred, y_lower, y_upper, _, _ = multi_output_data
        path = str(tmp_path / "test.png")
        plot_conformal_intervals_ordered(
            y_true, y_lower, y_upper, y_pred=y_pred, savefig=path
        )


class TestPlotConformalCoverage:
    def test_single_output(self, single_output_data):
        y_true, y_pred, _, _, _, residuals = single_output_data
        plot_conformal_coverage(
            y_true.reshape(-1, 1),
            y_pred.reshape(-1, 1),
            residuals.reshape(-1, 1),
        )

    def test_multi_output(self, multi_output_data):
        y_true, y_pred, _, _, _, residuals = multi_output_data
        plot_conformal_coverage(y_true, y_pred, residuals)


class TestPlotConformalVsGaussian:
    def test_multi_output(self, multi_output_data):
        y_true, y_pred, _, _, y_std, residuals = multi_output_data
        plot_conformal_vs_gaussian(y_true, y_pred, y_std, residuals)


class TestPlotConditionalCoverage:
    def test_single_output(self, single_output_data, tmp_path):
        y_true, _, y_lower, y_upper, _, _ = single_output_data
        path = str(tmp_path / "test.png")
        plot_conditional_coverage(y_true, y_lower, y_upper, savefig=path)

    def test_multi_output(self, multi_output_data, tmp_path):
        y_true, _, y_lower, y_upper, _, _ = multi_output_data
        path = str(tmp_path / "test.png")
        plot_conditional_coverage(y_true, y_lower, y_upper, savefig=path)
