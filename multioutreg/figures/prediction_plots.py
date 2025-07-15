# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_predictions(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.DataFrame | np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of predicted vs. true values.

    Works for both single-output and multi-output regressors and mirrors
    the visualisation used in scikit-learn's multioutput regression
    example.
    """

    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

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


def plot_predictions_with_error_bars(
    y_true, preds, std, output_names=None, n_cols=3
):
    """
    Plot predicted vs true values with error bars for multi- or single-output regression.

    Parameters:
    - y_true: shape (n_samples, n_outputs) or (n_samples,)
    - preds: shape (n_samples, n_outputs) or (n_samples,)
    - std:  shape (n_samples, n_outputs) or (n_samples,) standard deviation of prediction per output
    - output_names: list of names for output variables (optional)
    - n_cols: number of subplot columns (default: 3)
    """
    # Reshape to 2D if input is 1D
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    std  = np.asarray(std)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        std  = std.reshape(-1, 1)

    n_targets = y_true.shape[1]

    # Use provided names or default
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    # Handle single output
    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.errorbar(
            y_true[:, 0], preds[:, 0], yerr=std[:, 0],
            fmt='o', alpha=0.5, ecolor='gray', capsize=2
        )
        min_v = min(y_true[:, 0].min(), preds[:, 0].min())
        max_v = max(y_true[:, 0].max(), preds[:, 0].max())
        ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.set_title(output_names[0])
        plt.tight_layout()
        plt.show()
        return

    # Multi-output
    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        axs[i].errorbar(
            y_true[:, i], preds[:, i], yerr=std[:, i], fmt='o', alpha=0.5, ecolor='gray', capsize=2
        )
        min_v = min(y_true[:, i].min(), preds[:, i].min())
        max_v = max(y_true[:, i].max(), preds[:, i].max())
        axs[i].plot(
            [min_v, max_v], [min_v, max_v], 'k--', lw=2
        )
        axs[i].set_xlabel("True values")
        axs[i].set_ylabel("Predicted values")
        axs[i].set_title(output_names[i])

    # Hide any unused subplots
    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()
