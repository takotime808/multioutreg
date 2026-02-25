# Copyright (c) 2025 takotime808

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest

from multioutreg.time_series.chronos_adapter import ForecastResult
from multioutreg.time_series.figures import plot_forecast_result


def _make_result(n_quantiles=3, horizon=8):
    """Create a synthetic ForecastResult for testing."""
    q = np.zeros((1, n_quantiles, horizon))
    # lower=0, median=5, upper=10
    q[0, 0, :] = 0.0
    q[0, 1, :] = 5.0
    q[0, 2, :] = 10.0
    q_levels = (0.1, 0.5, 0.9)[:n_quantiles]
    return ForecastResult(quantiles=q, q_levels=q_levels, ids=("y",))


class TestPlotForecastResult:

    def test_returns_figure(self):
        res = _make_result()
        fig = plot_forecast_result(res)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_history(self):
        res = _make_result()
        history = np.linspace(0, 5, 20)
        fig = plot_forecast_result(res, history=history)
        assert fig is not None
        plt.close(fig)

    def test_custom_axes(self):
        res = _make_result()
        fig_ext, ax = plt.subplots()
        fig_out = plot_forecast_result(res, ax=ax)
        assert fig_out is fig_ext
        plt.close(fig_out)

    def test_title_in_figure(self):
        res = _make_result()
        fig = plot_forecast_result(res, title="My Forecast")
        titles = [ax.get_title() for ax in fig.axes]
        assert any("My Forecast" in t for t in titles)
        plt.close(fig)

    def test_single_quantile_no_error(self):
        """Even with just one quantile (no fan) the function should not crash."""
        q = np.array([[[1.0, 2.0, 3.0]]])
        res = ForecastResult(quantiles=q, q_levels=(0.5,), ids=("y",))
        fig = plot_forecast_result(res)
        assert fig is not None
        plt.close(fig)

    def test_series_idx_selects_correct_series(self):
        """With two series, series_idx=1 should use the second."""
        q = np.array([
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],  # series 0
            [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],  # series 1
        ])
        res = ForecastResult(quantiles=q, q_levels=(0.1, 0.5, 0.9), ids=("a", "b"))
        fig = plot_forecast_result(res, series_idx=1)
        assert fig is not None
        plt.close(fig)
