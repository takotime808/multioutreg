# Copyright (c) 2025 takotime808

import numpy as np
# import uncertainty_toolbox as uct
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Tuple, Union


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    ax: Union[Axes, None] = None,
    target: str = "output",
) -> Axes:
    """
    Plot residuals (Observed - Predicted) for each target in a multi-output regression problem.

    Args:
        y_pred: Predicted means (1D array).
        y_std: Predicted stds (1D array).
        y_true: True labels (1D array).
        n_subset: Number of points to randomly select.
        ylims: Optional y-axis limits.
        num_stds_confidence_bound: Interval width (number of stds).
        ax: matplotlib axes (optional).
        target: Name of target variable.

    Returns:
        Axes: The plot axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    if n_subset is not None and len(y_pred) > n_subset:
        idx = np.random.choice(len(y_pred), n_subset, replace=False)
        y_pred, y_std, y_true = y_pred[idx], y_std[idx], y_true[idx]
    order = np.argsort(y_true)
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(y_pred))
    intervals = num_stds_confidence_bound * y_std
    ax.errorbar(xs, y_pred, intervals, fmt="o", ls="none", linewidth=1.5, c="#1f77b4", alpha=0.5, label="Predicted Values")
    ax.plot(xs, y_pred, "o", c="#1f77b4", label=None)
    ax.plot(xs, y_true, "--", linewidth=2.0, c="#ff7f0e", label="Observed Values")
    if ylims is not None:
        ax.set_ylim(ylims)
    ax.set_title(target, fontsize=14)
    ax.tick_params(labelsize=10)
    return ax


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
    target_list: list = None,
    ylims: Union[Tuple[float, float], None] = None,
) -> List[Axes]:
    """
    Plot ordered prediction intervals for each dimension (output) of a multi-output regression problem.

    This function generates a separate subplot for each output dimension, plotting the predicted means,
    confidence intervals (as error bars), and observed values, ordered by the observed value. X and Y labels
    are shared across subplots. Optionally, only a random subset of samples can be plotted.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted means, shape (n_samples, n_outputs) or (n_samples,).
    y_std : np.ndarray
        Predicted standard deviations, same shape as `y_pred`.
    y_true : np.ndarray
        True target values, same shape as `y_pred`.
    ax_list : list of matplotlib.axes.Axes or None, optional
        List of matplotlib axes to plot on. If None, new axes are created.
    n_subset : int or None, optional
        If provided, plot only this many samples (randomly selected).
    num_stds_confidence_bound : int, optional
        Number of standard deviations to use for error bars (default: 2).
    savefig : str or bool, optional
        If a string, the path to save the resulting figure. If False (default), figure is not saved.
    suptitle : str, optional
        Supertitle for the full figure (default: "Ordered Prediction Intervals per Output").
    supxlabel : str, optional
        Shared x-axis label for the figure.
    supylabel : str, optional
        Shared y-axis label for the figure.
    target_list : list of str or None, optional
        List of output (target) names, one per output dimension. Used as subplot titles.
        If None, targets are named "Output {i}" by default.
    ylims : tuple of float or None, optional
        Shared y-axis limits for all subplots as (lower, upper).

    Returns
    -------
    list of matplotlib.axes.Axes
        List of axes objects corresponding to each output dimension.

    Raises
    ------
    ValueError
        If the input arrays do not have compatible shapes or dimensions.

    Examples
    --------
    >>> axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    >>> axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, n_subset=50, target_list=["Y1", "Y2"])

    Notes
    -----
    Each subplot visualizes:
      - predicted means (with error bars showing uncertainty),
      - observed (true) values,
      - shared axis labels and an overall figure title.
    """
    y_pred = np.asarray(y_pred)
    y_std = np.asarray(y_std)
    y_true = np.asarray(y_true)

    if y_pred.ndim == 1:
        if target_list is None:
            target_list = ["Output"]
        ax = plot_intervals_ordered(
            y_pred,
            y_std,
            y_true,
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax_list[0] if isinstance(ax_list, (list, np.ndarray)) else ax_list,
            target=target_list[0],
        )
        fig = plt.gcf()
        fig.supxlabel(supxlabel, fontsize=16)
        fig.supylabel(supylabel, fontsize=16)
        plt.suptitle(suptitle, fontsize=18, y=1.04)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if savefig:
            plt.savefig(savefig)
            plt.close()
        else:
            plt.show()
        return [ax]

    if not (y_pred.ndim == y_std.ndim == y_true.ndim == 2):
        raise ValueError("y_pred, y_std, and y_true must be 2D arrays")
    if not (y_pred.shape == y_std.shape == y_true.shape):
        raise ValueError("y_pred, y_std, and y_true must have the same shape")

    n_outputs = y_pred.shape[1]
    if target_list is None:
        target_list = [f"Output {i+1}" for i in range(n_outputs)]
    if ax_list is None:
        fig, ax_list = plt.subplots(1, n_outputs, figsize=(5 * n_outputs, 5))
    else:
        fig = plt.gcf()

    axes = np.atleast_1d(ax_list).ravel()
    out_axes = []
    for i in range(n_outputs):
        ax_i = axes[i]
        ax_i.set_xlabel("")
        ax_i.set_ylabel("")
        ax_i = plot_intervals_ordered(
            y_pred[:, i],
            y_std[:, i],
            y_true[:, i],
            n_subset=n_subset,
            ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=ax_i,
            target=target_list[i],
        )
        out_axes.append(ax_i)

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
