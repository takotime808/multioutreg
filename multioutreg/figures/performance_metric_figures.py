# Copyright (c) 2025 takotime808

import matplotlib.pyplot as plt
import pandas as pd
from typing import Iterable, Optional
import numpy as np


def plot_uq_metrics_bar(
    metrics_df: pd.DataFrame,
    metrics: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot bar chart of per-output uncertainty metrics.

    Parameters
    ----------
    metrics_df:
        DataFrame returned by ``get_uq_performance_metrics_flexible`` containing
        one row per output.
    metrics:
        Iterable of metric column names to plot. If ``None``, the function will
        try to plot commonly used metrics such as ``rmse`` and ``mae`` if they
        are present in ``metrics_df``.
    ax:
        Optional :class:`matplotlib.axes.Axes` to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the bar plot.
    """
    if "accuracy" in metrics_df.columns and isinstance(metrics_df.loc[0, "accuracy"], dict):
        # Expand nested dictionary columns returned by ``get_uq_performance_metrics_flexible``
        expanded_rows = []
        for _, row in metrics_df.iterrows():
            data = {}
            for col in ["accuracy", "avg_calibration", "scoring_rule", "sharpness"]:
                if col in row and isinstance(row[col], dict):
                    data.update(row[col])
            data["output"] = row["output"]
            expanded_rows.append(data)
        metrics_df = pd.DataFrame(expanded_rows)

    if metrics is None:
        default_metrics = ["rmse", "mae", "nll", "miscal_area"]
        metrics = [m for m in default_metrics if m in metrics_df.columns]
    else:
        metrics = [m for m in metrics if m in metrics_df.columns]

    if not metrics:
        raise ValueError(
            f"None of the requested metrics are available. Columns: {list(metrics_df.columns)}"
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    metrics_df[metrics].plot.bar(ax=ax)
    ax.set_xticklabels([f"Output {i}" for i in metrics_df["output"]])
    ax.set_xlabel("Output")
    ax.set_title("Uncertainty Toolbox Metrics per Output")
    ax.legend(title="Metric")
    plt.tight_layout()
    return ax


