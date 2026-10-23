# Copyright (c) 2025 takotime808

from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from multioutreg.time_series.chronos_adapter import ForecastResult


def plot_forecast_result(
    result: ForecastResult,
    history: Optional[np.ndarray] = None,
    series_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> plt.Figure:
    """Plot a ForecastResult for one series as a quantile fan chart.

    Plots the median quantile as the point forecast and shades the outermost
    quantile band as a confidence fan.  If ``history`` is provided, the
    historical values are prepended on the left side.

    Parameters
    ----------
    result : ForecastResult
        Must have shape ``[n_series, n_quantiles, horizon]``.
    history : np.ndarray | None
        1D array of historical observations to show before the forecast.
    series_idx : int, default 0
        Which series in ``result.ids`` to plot.
    ax : plt.Axes | None
        Axes to plot into; a new figure is created when ``None``.
    title : str
        Plot title.

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    q_arr = result.quantiles[series_idx]  # (Q, H)
    q_levels = result.q_levels
    horizon = q_arr.shape[1]

    # Identify median and outer band
    med_idx = None
    if 0.5 in q_levels:
        med_idx = q_levels.index(0.5)

    # x positions: history on the left (negative), forecast on the right (0+)
    n_hist = len(history) if history is not None else 0
    x_hist = np.arange(-n_hist, 0)
    x_fore = np.arange(0, horizon)

    if history is not None:
        ax.plot(x_hist, np.asarray(history).ravel(), color="steelblue", label="History")
        # Connect the last history point to the first forecast point
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Shade fan: outer quantile band
    if len(q_levels) >= 2:
        lower = q_arr[0]
        upper = q_arr[-1]
        ax.fill_between(x_fore, lower, upper, alpha=0.25, color="orange", label="Forecast interval")

    # Plot median (or mean as fallback)
    if med_idx is not None:
        median = q_arr[med_idx]
    else:
        median = q_arr.mean(axis=0)
    ax.plot(x_fore, median, color="orange", linewidth=1.8, label="Forecast (median)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(title or "Forecast")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
