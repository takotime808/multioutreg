# Copyright (c) 2025 takotime808

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
from typing import List


def plot_residuals_multioutput(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    max_cols: int = 3,
    base_width: int = 8,
    base_height: int = 7,
    savefig: str = False,
    suptitle: str = "Residuals vs. Predicted Value",
    supxlabel: str = "Predicted Value",
    supylabel: str = "Residual (Observed - Predicted)",
    target_list: list = None,
) -> List[plt.Axes]:
    """
    Plot residuals (Observed - Predicted) for each target in a multi-output regression problem.

    Each output (target) gets its own subplot, all sharing y-limits for easy visual comparison.
    Layout is a grid with up to `max_cols` columns per row. Shared axis labels and a customizable
    figure title are supported.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets) or (n_samples,).
    y_true : np.ndarray
        True values, shape (n_samples, n_targets) or (n_samples,).
    max_cols : int, optional
        Maximum number of columns in the subplot grid (default is 3).
    base_width : int, optional
        Width of each subplot in inches (default is 8).
    base_height : int, optional
        Height of each subplot in inches (default is 7).
    savefig : str or bool, optional
        If a string, the path where the figure should be saved (e.g. "plot.png"). If False (default), the plot is shown instead.
    suptitle : str, optional
        Overall title for the figure (default is "Residuals vs. Predicted Value").
    supxlabel : str, optional
        Shared x-axis label for all subplots (default is "Predicted Value").
    supylabel : str, optional
        Shared y-axis label for all subplots (default is "Residual (Observed - Predicted)").
    target_list : list of str, optional
        List of titles for each target subplot (default is ["Target 1", "Target 2", ...]).

    Returns
    -------
    List[plt.Axes]
        List of matplotlib Axes objects for each subplot.

    Example
    -------
    >>> plot_residuals_multioutput(y_pred, y_true, suptitle="Residuals for Each Output")
    >>> plot_residuals_multioutput(y_pred, y_true, savefig="residuals.png", target_list=["Pressure", "Temperature"])

    Notes
    -----
    - If `savefig` is provided as a string, the figure is saved to disk and not displayed interactively.
    - Extra/empty subplot axes are hidden if n_targets is not a multiple of max_cols.
    """
    y_pred = np.atleast_2d(y_pred)
    y_true = np.atleast_2d(y_true)
    if y_pred.shape[0] == 1 or y_pred.shape[0] != y_true.shape[0]:
        y_pred = y_pred.T
        y_true = y_true.T
    residuals = y_true - y_pred

    n_targets = y_pred.shape[1]
    n_cols = min(n_targets, max_cols)
    n_rows = ceil(n_targets / n_cols)

    # Make target list for plot sup titles
    if target_list is None:
        target_list = [f"Target {i+1}" for i in range(n_targets)]
    
    # Shared y-axis for all
    rmin = np.floor(np.min(residuals))
    rmax = np.ceil(np.max(residuals))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(base_width * n_cols, base_height * n_rows),
        sharey=True,
        constrained_layout=True
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()
    for i in range(n_targets):
        ax = axes_flat[i]
        ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.7, color="#1f77b4", edgecolor="k")
        ax.axhline(0, color="red", linestyle="--", lw=2)
        ax.set_ylim([rmin, rmax])
        ax.set_title(target_list[i], fontsize=18)
        ax.tick_params(labelsize=12)

    # Hide unused axes
    for j in range(n_targets, n_rows * n_cols):
        axes_flat[j].axis('off')

    # Shared labels
    fig.supxlabel(supxlabel, fontsize=18)
    fig.supylabel(supylabel, fontsize=18)
    fig.suptitle(suptitle + f" (n_targets={n_targets})", fontsize=22)

    axes_flat[:n_targets]

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()

    return axes_flat[:n_targets]


def plot_residuals_multioutput_with_regplot(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    max_cols: int = 3,
    base_width: int = 8,
    base_height: int = 7,
    savefig: str = False,
    suptitle: str = "Residuals vs. Predicted Value",
    supxlabel: str = "Predicted Value",
    supylabel: str = "Residual (Observed - Predicted)",
    target_list: list = None,
) -> List[plt.Axes]:
    """
    Plot residuals (Observed - Predicted) for each target in a multi-output regression problem,
    using seaborn.regplot (scatter+trend line) in each subplot.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets) or (n_samples,).
    y_true : np.ndarray
        True values, shape (n_samples, n_targets) or (n_samples,).
    max_cols : int, optional
        Maximum number of columns in the subplot grid (default is 3).
    base_width : int, optional
        Width of each subplot in inches (default is 8).
    base_height : int, optional
        Height of each subplot in inches (default is 7).
    savefig : str or bool, optional
        If a string, the path where the figure should be saved (e.g. "plot.png"). If False (default), the plot is shown instead.
    suptitle : str, optional
        Overall title for the figure (default is "Residuals vs. Predicted Value").
    supxlabel : str, optional
        Shared x-axis label for all subplots (default is "Predicted Value").
    supylabel : str, optional
        Shared y-axis label for all subplots (default is "Residual (Observed - Predicted)").
    target_list : list of str, optional
        List of titles for each target subplot (default is ["Target 1", "Target 2", ...]).

    Returns
    -------
    List[plt.Axes]
        List of matplotlib Axes objects for each subplot.
    """
    y_pred = np.atleast_2d(y_pred)
    y_true = np.atleast_2d(y_true)
    if y_pred.shape[0] == 1 or y_pred.shape[0] != y_true.shape[0]:
        y_pred = y_pred.T
        y_true = y_true.T
    residuals = y_true - y_pred

    n_targets = y_pred.shape[1]
    n_cols = min(n_targets, max_cols)
    n_rows = ceil(n_targets / n_cols)

    if target_list is None:
        target_list = [f"Target {i+1}" for i in range(n_targets)]
    
    # rmin = np.floor(np.min(residuals)) # NOTE: un-comment to revert.
    # rmax = np.ceil(np.max(residuals)) # NOTE: un-comment to revert.
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(base_width * n_cols, base_height * n_rows),
        # sharey=True, # NOTE: un-comment to revert.
        sharey=False, # NOTE: comment to revert.
        sharex=False, # NOTE: comment to revert.
        constrained_layout=True
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()
    for i in range(n_targets):
        yres = residuals[:, i] # NOTE: comment to revert.
        yp = y_pred[:, i] # NOTE: comment to revert.
        ax = axes_flat[i]
        sns.regplot(
            x=y_pred[:, i], y=residuals[:, i], ax=ax,
            scatter_kws={'alpha': 0.7, 's': 40}, line_kws={'color': 'black', 'lw': 2}
        )
        ax.axhline(0, color="red", linestyle="--", lw=2)
        # ax.set_ylim([rmin, rmax]) # NOTE: un-comment to revert.
        margin = 0.1 * (np.max(yres) - np.min(yres) + 1e-8)  # avoid zero range # NOTE: comment to revert.
        ax.set_ylim(np.min(yres) - margin, np.max(yres) + margin) # NOTE: comment to revert.
        ax.set_title(target_list[i], fontsize=18)
        ax.tick_params(labelsize=12)
    for j in range(n_targets, n_rows * n_cols):
        axes_flat[j].axis('off')
    fig.supxlabel(supxlabel, fontsize=18)
    fig.supylabel(supylabel, fontsize=18)
    fig.suptitle(suptitle + f" (n_targets={n_targets})", fontsize=22)

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()

    return axes_flat[:n_targets]