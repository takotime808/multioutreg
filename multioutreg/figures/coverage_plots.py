# Copyright (c) 2025 takotime808 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_coverage(
    y_true, y_pred, y_std, output_names=None, n_cols=3,
    intervals = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99],
):
    """
    Coverage plot: for each confidence level, plots nominal vs empirical coverage.

    Parameters:
        y_true, y_pred, y_std: arrays, shapes (n_samples, n_outputs) or (n_samples,)
        output_names: list of output variable names (optional)
        n_cols: subplots columns if multi-output
        intervals: list of confidence intervals (as fractions, e.g. 0.95 for 95%)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_std  = np.asarray(y_std)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        y_std  = y_std.reshape(-1, 1)

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    # Compute coverage for each output and each interval
    empirical = []
    for i in range(n_targets):
        cov = []
        for ci in intervals:
            z = norm.ppf(0.5 + ci/2)  # two-sided
            lower = y_pred[:, i] - z * y_std[:, i]
            upper = y_pred[:, i] + z * y_std[:, i]
            frac = np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper))
            cov.append(frac)
        empirical.append(cov)

    empirical = np.array(empirical)

    # Plot
    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(intervals, empirical[0], 'o-', label='Empirical coverage')
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
        ax.set_xlabel("Nominal confidence level")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(output_names[0])
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axs = np.array(axs).reshape(-1)
    for i in range(n_targets):
        axs[i].plot(intervals, empirical[i], 'o-', label='Empirical coverage')
        axs[i].plot([0, 1], [0, 1], 'k--', label='Ideal')
        axs[i].set_xlabel("Nominal confidence level")
        axs[i].set_ylabel("Empirical coverage")
        axs[i].set_title(output_names[i])
        axs[i].legend()
    for j in range(n_targets, n_rows*n_cols):
        axs[j].set_visible(False)
    plt.tight_layout()
    plt.show()
