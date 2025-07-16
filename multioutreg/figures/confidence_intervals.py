# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from scipy.stats import norm
from typing import Optional, List, Union, Tuple
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Optional[int] = None,
    ylims: Optional[Tuple[float, float]] = None,
    num_stds_confidence_bound: float = 2,
    ax: Optional[plt.Axes] = None,
    add_legend: bool = True,
) -> Tuple[plt.Line2D, plt.Line2D, plt.Axes]:
    """
    Plot ordered predicted values and intervals for a single target.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, shape (n_samples,).
    y_std : np.ndarray
        Predicted standard deviations, shape (n_samples,).
    y_true : np.ndarray
        Observed/true values, shape (n_samples,).
    n_subset : int, optional
        If provided, subsample n_subset random points.
    ylims : tuple of float, optional
        Y-axis limits as (ymin, ymax).
    num_stds_confidence_bound : float, optional
        Number of standard deviations for error bars. Default is 2.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, a new one is created.
    add_legend : bool, optional
        Whether to add a legend to the plot.

    Returns
    -------
    pred_line : plt.Line2D
        The line artist for the predicted values.
    obs_line : plt.Line2D
        The line artist for the observed values.
    ax : plt.Axes
        The matplotlib axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    if n_subset is not None:
        idx = np.random.choice(len(y_pred), n_subset, replace=False)
        idx = np.sort(idx)
        y_pred, y_std, y_true = y_pred[idx], y_std[idx], y_true[idx]
    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std
    pred_points = ax.errorbar(xs, y_pred, intervals, fmt="o", ls="none", linewidth=1.5, c="#1f77b4", alpha=0.5, label="Predicted Values")
    pred_line, = ax.plot(xs, y_pred, "o", c="#1f77b4", label=None)
    obs_line, = ax.plot(xs, y_true, "--", linewidth=2.0, c="#ff7f0e", label="Observed Values")
    if add_legend:
        ax.legend([pred_line, obs_line], ["Predicted Values", "Observed Values"], loc=4)
    if ylims is not None:
        ax.set_ylim(ylims)
    # Remove per-axis labels for shared labels
    # ax.set_xlabel("Index (Ordered by Observed Value)", fontsize=14)
    # ax.set_ylabel("Predicted Values and Intervals", fontsize=14)
    ax.set_title("Ordered Prediction Intervals", fontsize=18)
    ax.tick_params(labelsize=12)
    return pred_line, obs_line, ax


def plot_intervals_ordered_multi(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    handles: Optional[List[plt.Line2D]] = None,
    n_subset: Optional[int] = None,
    base_height: int = 7,
    base_width: int = 8,
    num_stds_confidence_bound: float = 2,
    max_cols: int = 3,
    savefig: Union[str, bool] = False,
    supxlabel: str = "Index (Ordered by Observed Value)",
    supylabel: str = "Predicted Values and Intervals",
    suptitle: str = "Ordered Prediction Intervals",
    target_list: Optional[List[str]] = None,
) -> None:
    """
    Plot ordered prediction intervals for multi-output regression.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_targets) or (n_targets, n_samples).
    y_std : np.ndarray
        Predicted standard deviations, shape (n_samples, n_targets) or (n_targets, n_samples).
    y_true : np.ndarray
        Observed/true values, shape (n_samples, n_targets) or (n_targets, n_samples).
    handles : list of plt.Line2D, optional
        Handles for legend entries.
    n_subset : int, optional
        Subsample to this many data points.
    base_height : int, optional
        Height per subplot, in inches.
    base_width : int, optional
        Width per subplot, in inches.
    num_stds_confidence_bound : float, optional
        Number of standard deviations for intervals.
    max_cols : int, optional
        Maximum columns in grid layout.
    savefig : str or bool, optional
        Path to save the figure. If False, shows the plot.
    supxlabel : str, optional
        Super x-axis label for the figure.
    supylabel : str, optional
        Super y-axis label for the figure.
    suptitle : str, optional
        Super title for the figure.
    target_list : list of str, optional
        Names of targets to use as subplot titles.

    Returns
    -------
    None
    """
    y_pred = np.atleast_2d(y_pred)
    y_std = np.atleast_2d(y_std)
    y_true = np.atleast_2d(y_true)

    if y_pred.shape[0] == 1 or y_pred.shape[0] != y_true.shape[0]:
        y_pred = y_pred.T
        y_std = y_std.T
        y_true = y_true.T

    n_targets = y_pred.shape[1]

    if target_list is None:
        target_list = [f"Output {idx}" for idx in range(n_targets)]

    n_cols = min(n_targets, max_cols)
    n_rows = ceil(n_targets / n_cols)
    intervals = num_stds_confidence_bound * y_std

    ylims = (np.floor(np.min(y_pred - intervals)), np.ceil(np.max(y_pred + intervals)))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(base_width * n_cols, base_height * n_rows),
        sharey=True,
        constrained_layout=True
    )

    axes = np.array(axes).reshape(n_rows, n_cols)
    axes_flat = axes.flatten()

    for i in range(n_targets):
        pred_line, obs_line, ax = plot_intervals_ordered(
            y_pred[:, i], y_std[:, i], y_true[:, i],
            n_subset=n_subset, ylims=ylims,
            num_stds_confidence_bound=num_stds_confidence_bound,
            ax=axes_flat[i], add_legend=False
        )
        if handles is None:
            handles = [pred_line, obs_line]
        axes_flat[i].set_title(f"Target {i+1}", fontsize=18)
        axes_flat[i].tick_params(labelsize=12)

    for j in range(n_targets, n_rows * n_cols):
        axes_flat[j].axis('off')

    # # Shared legend at the bottom
    # fig.legend(
    #     handles,
    #     ["Predicted Values", "Observed Values"],
    #     loc='lower center',
    #     ncol=2,
    #     fontsize=16,
    #     frameon=False,
    #     bbox_to_anchor=(0.5, 0.01)
    # )

    # Shared legend at the top left
    fig.legend(
        handles, ["Predicted Values", "Observed Values"],
        loc='upper left',
        bbox_to_anchor=(0.0001, 0.9999),
        fontsize=16,
        frameon=False
    )

    fig.supxlabel(supxlabel, fontsize=18)
    fig.supylabel(supylabel, fontsize=18)
    fig.suptitle(suptitle + f" (n_targets={n_targets})", fontsize=22)
    plt.subplots_adjust(bottom=0.13)

    axes_flat[:n_targets]

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    # return axes_flat[:n_targets]
    return


def plot_confidence_interval(
    y_true: Union[np.ndarray, List[float]],
    preds: Union[np.ndarray, List[float]],
    std: Union[np.ndarray, List[float]],
    output_names: Optional[List[str]] = None,
    n_cols: int = 3,
    alpha: float = 0.05,
    plot_errorbars: bool = False,
    plot_ci_band: bool = True,
    savefig: Optional[str] = None,
) -> None:
    """
    Plot predicted values with confidence intervals and error bars.

    Handles single and multi-output regression, with error bars on predictions.

    Parameters
    ----------
    y_true : np.ndarray or list of float
        Ground truth values, shape (n_samples, n_outputs) or (n_samples,).
    preds : np.ndarray or list of float
        Predicted mean values, shape (n_samples, n_outputs) or (n_samples,).
    std : np.ndarray or list of float
        Predicted standard deviations, shape (n_samples, n_outputs) or (n_samples,).
    output_names : list of str, optional
        Names for output variables. Default is ["Output 0", ...].
    n_cols : int, optional
        Number of subplot columns for multi-output. Default is 3.
    alpha : float, optional
        Significance level for confidence interval (default 0.05 for 95% CI).
    plot_errorbars : bool, optional
        Option to include errorbars.
    plot_ci_band: bool, optional
        Option to include confidence interval.
    savefig : str or None, optional
        Path to save figure. If None, the figure is shown. Default is None.

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    std  = np.asarray(std)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
        std  = std.reshape(-1, 1)

    n_targets = y_true.shape[1]
    z = abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - alpha / 2)))  # ~1.96 for 95%

    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    x = np.arange(len(preds))

    # Single output case
    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        lower = preds[:, 0] - z * std[:, 0]
        upper = preds[:, 0] + z * std[:, 0]
        # True values
        ax.plot(x, y_true[:, 0], label='True', marker='o', linestyle='None', alpha=0.6)

        # Predicted means with error bars
        if plot_errorbars:
            ax.errorbar(
                x, preds[:, 0], yerr=z * std[:, 0],
                fmt='x', color='C1', ecolor='C1', elinewidth=1.5, capsize=3, label='Predicted ± CI'
            )
        else:
            ax.plot(x, preds[:, 0], color='C1', label='Predicted', marker='o', linestyle='None', alpha=0.6)

        # CI band
        if plot_ci_band:
            ax.fill_between(
                x,
                lower,
                upper,
                color="gray",
                alpha=0.3,
                label=f'{int((1-alpha)*100)}% CI'
            )

        ax.set_xlabel("Sample index")
        ax.set_ylabel("Value")
        ax.set_title(output_names[0])
        ax.legend()
        plt.tight_layout()
        if savefig:
            plt.savefig(savefig)
            plt.close()
        else:
            plt.show()
        return

    # Multi-output
    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        lower = preds[:, i] - z * std[:, i]
        upper = preds[:, i] + z * std[:, i]
        axs[i].plot(x, y_true[:, i], label='True', marker='o', linestyle='None', alpha=0.6)
        if plot_errorbars:
            axs[i].errorbar(
                x, preds[:, i], yerr=z * std[:, i],
                fmt='x', color='C1', ecolor='C1', elinewidth=1.5, capsize=3, label='Predicted ± CI'
            )
        else:
            axs[i].plot(x, preds[:, i], label='Predicted', marker='o', linestyle='None', alpha=0.6)

        if plot_ci_band:
            axs[i].fill_between(
                x,
                lower,
                upper,
                color="gray",
                alpha=0.3,
                label=f'{int((1-alpha)*100)}% CI'
            )
        axs[i].set_xlabel("Sample index")
        axs[i].set_ylabel("Value")
        axs[i].set_title(output_names[i])
        axs[i].legend()
    # Hide unused axes
    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return


def plot_multioutput_confidence_intervals(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    base_estimator: Optional[object] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    plot_errorbars: bool = False,
    plot_ci_band: bool = True,
    savefig: Optional[str] = None,
) -> None:
    """
    Fit a MultiOutputRegressor and plot predictions with confidence intervals and error bars.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target matrix of shape (n_samples, n_targets).
    alpha : float, optional
        Significance level for confidence interval (default 0.05 for 95% CI).
    base_estimator : sklearn.base.RegressorMixin, optional
        Regressor to use in MultiOutputRegressor (default: RandomForestRegressor).
    test_size : float, optional
        Fraction of data to use for testing. Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    plot_errorbars : bool, optional
        Option to include errorbars.
    plot_ci_band: bool, optional
        Option to include confidence interval.
    savefig : str or None, optional
        Path to save figure. If None, the figure is shown. Default is None.

    Returns
    -------
    None
    """
    if base_estimator is None:
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_train, Y_train)
    
    means = []
    stds = []
    for est in model.estimators_:
        preds = np.array([tree.predict(X_test) for tree in est.estimators_])
        means.append(preds.mean(axis=0))
        stds.append(preds.std(axis=0))
    means = np.stack(means, axis=1)
    stds = np.stack(stds, axis=1)

    # Compute z-value for the specified alpha
    z = norm.ppf(1 - alpha / 2)

    n_targets = Y.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(8, 3 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]
    x_axis = np.arange(X_test.shape[0])

    for i in range(n_targets):
        ax = axes[i]
        # True values
        ax.plot(x_axis, Y_test[:, i], 'o', label='True')
        # Predicted mean line
        ax.plot(x_axis, means[:, i], '-', label='Predicted')

        # Error bars on predicted mean
        if plot_errorbars:
            ax.errorbar(
                x_axis, means[:, i], yerr=z * stds[:, i],
                fmt='x', color='C1', ecolor='C1', elinewidth=1.5, capsize=3, label='Predicted ± CI'
            )

        # CI band
        if plot_ci_band:
            ax.fill_between(
                x_axis,
                means[:, i] - z * stds[:, i],
                means[:, i] + z * stds[:, i],
                color='orange',
                alpha=0.3,
                label=f'{int((1-alpha)*100)}% CI' if i == 0 else None,
            )

        ax.set_title(f'Target {i+1}')
        ax.legend()
        ax.set_ylabel('Value')
    axes[-1].set_xlabel('Test Sample Index')
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return


def plot_regressormixin_confidence_intervals(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    base_estimator: Optional[RegressorMixin] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    plot_errorbars : bool = False, 
    plot_ci_band: bool = True, 
    savefig: str = False,
) -> None:
    """
    Fit a MultiOutputRegressor and plot predictions with confidence intervals and error bars.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target matrix of shape (n_samples, n_targets).
    alpha : float, optional
        Significance level for confidence interval (default 0.05 for 95% CI).
    base_estimator : sklearn.base.RegressorMixin, optional
        Regressor to use in MultiOutputRegressor (default: RandomForestRegressor).
    test_size : float, optional
        Fraction of data to use for testing. Default is 0.2.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    plot_errorbars : bool, optional
        Option to include errorbars.
    plot_ci_band: bool, optional
        Option to include confidence interval.
    savefig : str or None, optional
        Path to save figure. If None, the figure is shown. Default is None.

    Returns
    -------
    None
    """
    import numpy as np
    from scipy.stats import norm

    if base_estimator is None:
        base_estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    model = MultiOutputRegressor(base_estimator)
    model.fit(X_train, Y_train)
    
    means = []
    stds = []
    for est in model.estimators_:
        preds = np.array([tree.predict(X_test) for tree in est.estimators_])
        means.append(preds.mean(axis=0))
        stds.append(preds.std(axis=0))
    means = np.stack(means, axis=1)
    stds = np.stack(stds, axis=1)

    # Compute z-value for the specified alpha
    z = norm.ppf(1 - alpha / 2)

    n_targets = Y.shape[1]
    fig, axes = plt.subplots(n_targets, 1, figsize=(8, 3 * n_targets), sharex=True)
    if n_targets == 1:
        axes = [axes]
    x_axis = np.arange(X_test.shape[0])

    for i in range(n_targets):
        ax = axes[i]
        # Plot true values
        ax.plot(x_axis, Y_test[:, i], 'o', label='True')
        # Plot predicted mean as a line
        ax.plot(x_axis, means[:, i], '-', label='Predicted')

        # Plot error bars at each predicted point
        if plot_errorbars:
            ax.errorbar(
                x_axis, means[:, i], yerr=z * stds[:, i],
                fmt='x', color='C1', ecolor='C1', elinewidth=1.5, capsize=3, label='Predicted ± CI'
            )

        # Plot CI band for visual clarity
        if plot_ci_band:
            ax.fill_between(
                x_axis,
                means[:, i] - z * stds[:, i],
                means[:, i] + z * stds[:, i],
                color='orange',
                alpha=0.3,
                label=f'{int((1-alpha)*100)}% CI' if i == 0 else None,
            )

        ax.set_title(f'Target {i+1}')
        ax.legend()
        ax.set_ylabel('Value')
    axes[-1].set_xlabel('Test Sample Index')
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
    return
