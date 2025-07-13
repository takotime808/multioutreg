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


def plot_predictions(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.DataFrame | np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of predicted vs. true values for each output.

    This mirrors the visualisation used in scikit-learn's multioutput
    regression example.
    """
    import numpy as np

    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    n_outputs = y_true.shape[1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for i in range(n_outputs):
        ax.scatter(y_true[:, i], y_pred[:, i], s=15, label=f"Output {i}")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "k--", lw=2)
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title("Predicted vs. true values")
    ax.legend()
    plt.tight_layout()
    return ax