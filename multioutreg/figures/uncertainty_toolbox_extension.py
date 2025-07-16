# Copyright (c) 2025 takotime808

import numpy as np
import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Union


def plot_uct_intervals_ordered_multioutput(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    ax_list: Union[List[Axes], None] = None,
    n_subset: Union[int, None] = None,
    num_stds_confidence_bound: int = 2,
    savefig: str = False,
    suptitle: str = "Ordered Prediction Intervals per Output",
    supxlabel: str = "Index (Ordered by Observed Value)",
    supylabel: str = "Predicted Values and Intervals",
    ylims: Union[Tuple[float, float], None] = None,
) -> List[Axes]:
    """
    Plot ordered prediction intervals for multi-dimensional outputs.

    For each output dimension, this function plots the prediction intervals, ordered by the predicted means,
    using uncertainty_toolbox's `plot_intervals_ordered`. Each output gets its own subplot (axis).

    Args:
        y_pred (np.ndarray): Predicted means, shape (n_samples, n_outputs) or (n_samples,).
        y_std (np.ndarray): Predicted standard deviations, same shape as `y_pred`.
        y_true (np.ndarray): True target values, same shape as `y_pred`.
        ax_list (list[Axes] or None, optional): List of matplotlib axes to plot on. If None, new axes are created.
        n_subset (int or None, optional): If provided, plot only this many samples (randomly selected or filtered).
        num_stds_confidence_bound (int, optional): Number of standard deviations for confidence intervals (default: 2).
        savefig (str or bool, optional): If str, path to save the resulting figure. If False, does not save (default: False).
        suptitle (str, optional): Supertitle for the full figure (default: "Ordered Prediction Intervals per Output").
        supxlabel (str, optional): Shared x-axis label for the figure.
        supylabel (str, optional): Shared y-axis label for the figure.
        ylims (tuple[float, float] or None, optional): Y-axis limits for all subplots.

    Returns:
        list[Axes]: List of matplotlib axes objects, one per output dimension.

    Raises:
        ValueError: If input arrays have incompatible shapes or dimensions.

    Example:
        >>> axes = plot_intervals_ordered_multioutput(y_pred, y_std, y_true)
    """

    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    y_true = np.asarray(y_true)

    if y_pred.ndim == 1:
        ax = uct.plot_intervals_ordered(
            y_pred,
            y_std,
            y_true,
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax_list[0] if isinstance(ax_list, (list, np.ndarray)) else ax_list,
        )
        return [ax]

    if not (y_pred.ndim == y_std.ndim == y_true.ndim == 2):
        raise ValueError("y_pred, y_std, and y_true must be 2D arrays")

    if not (y_pred.shape == y_std.shape == y_true.shape):
        raise ValueError("y_pred, y_std, and y_true must have the same shape")

    n_outputs = y_pred.shape[1]

    if ax_list is None:
        fig, ax_list = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))

    axes = np.atleast_1d(ax_list).ravel()

    out_axes = []
    for i in range(n_outputs):
        ax_i = axes[i]
        ax_i = uct.plot_intervals_ordered(
            y_pred[:, i],
            y_std[:, i],
            y_true[:, i],
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax_i,
        )
        out_axes.append(ax_i)

    # Set and shared labels and suptitle
    fig.supxlabel(supxlabel, fontsize=16)
    fig.supylabel(supylabel, fontsize=16)
    plt.suptitle(suptitle, fontsize=18, y=1.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return out_axes
