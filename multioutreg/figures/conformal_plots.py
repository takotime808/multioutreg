# Copyright (c) 2025 takotime808

"""Visualization functions for conformal prediction intervals."""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from typing import List, Optional, Union

from multioutreg.conformal.base import BaseConformalPredictor


def plot_conformal_intervals(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    output_names: Optional[List[str]] = None,
    n_cols: int = 3,
    alpha: float = 0.1,
    savefig: Optional[str] = None,
) -> None:
    """Plot conformal prediction intervals alongside true values.

    Parameters
    ----------
    y_true : (n_samples,) or (n_samples, n_outputs)
    y_lower : same shape
    y_upper : same shape
    y_pred : same shape, optional. Point predictions.
    output_names : list of str, optional
    n_cols : int
    alpha : float, for labeling
    savefig : str, optional
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_lower = np.atleast_2d(np.asarray(y_lower))
    y_upper = np.atleast_2d(np.asarray(y_upper))
    if y_true.shape[0] == 1:
        y_true, y_lower, y_upper = y_true.T, y_lower.T, y_upper.T
    if y_pred is not None:
        y_pred = np.atleast_2d(np.asarray(y_pred))
        if y_pred.shape[0] == 1:
            y_pred = y_pred.T

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    x = np.arange(y_true.shape[0])
    ci_label = f"{int((1 - alpha) * 100)}% Conformal PI"

    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(x, y_true[:, 0], 'o', alpha=0.6, label="True")
        if y_pred is not None:
            ax.plot(x, y_pred[:, 0], 'o', alpha=0.6, label="Predicted")
        ax.fill_between(x, y_lower[:, 0], y_upper[:, 0],
                        color="gray", alpha=0.3, label=ci_label)
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

    n_rows = ceil(n_targets / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        axs[i].plot(x, y_true[:, i], 'o', alpha=0.6, label="True")
        if y_pred is not None:
            axs[i].plot(x, y_pred[:, i], 'o', alpha=0.6, label="Predicted")
        axs[i].fill_between(x, y_lower[:, i], y_upper[:, i],
                            color="gray", alpha=0.3, label=ci_label)
        axs[i].set_xlabel("Sample index")
        axs[i].set_ylabel("Value")
        axs[i].set_title(output_names[i])
        axs[i].legend()

    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()


def plot_conformal_intervals_ordered(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    output_names: Optional[List[str]] = None,
    n_cols: int = 3,
    alpha: float = 0.1,
    savefig: Optional[str] = None,
) -> None:
    """Plot conformal intervals ordered by true value.

    Parameters
    ----------
    y_true : (n_samples,) or (n_samples, n_outputs)
    y_lower, y_upper : same shape
    y_pred : optional, same shape
    output_names : list of str, optional
    n_cols : int
    alpha : float
    savefig : str, optional
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_lower = np.atleast_2d(np.asarray(y_lower))
    y_upper = np.atleast_2d(np.asarray(y_upper))
    if y_true.shape[0] == 1:
        y_true, y_lower, y_upper = y_true.T, y_lower.T, y_upper.T
    if y_pred is not None:
        y_pred = np.atleast_2d(np.asarray(y_pred))
        if y_pred.shape[0] == 1:
            y_pred = y_pred.T

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    ci_label = f"{int((1 - alpha) * 100)}% Conformal PI"

    n_cols_actual = min(n_targets, n_cols)
    n_rows = ceil(n_targets / n_cols_actual)
    fig, axs = plt.subplots(
        n_rows, n_cols_actual,
        figsize=(7 * n_cols_actual, 6 * n_rows),
        constrained_layout=True,
    )
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        order = np.argsort(y_true[:, i])
        xs = np.arange(len(order))

        axs[i].fill_between(
            xs, y_lower[order, i], y_upper[order, i],
            color="#1f77b4", alpha=0.3, label=ci_label,
        )
        axs[i].plot(xs, y_true[order, i], "--", linewidth=2.0,
                     c="#ff7f0e", label="Observed Values")
        if y_pred is not None:
            axs[i].plot(xs, y_pred[order, i], "o", c="#1f77b4",
                         alpha=0.5, label="Predicted Values")

        axs[i].set_title(output_names[i], fontsize=14)
        axs[i].legend()

    for j in range(n_targets, n_rows * n_cols_actual):
        axs[j].set_visible(False)

    fig.supxlabel("Index (Ordered by Observed Value)", fontsize=14)
    fig.supylabel("Value", fontsize=14)
    fig.suptitle("Ordered Conformal Prediction Intervals", fontsize=16)

    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()


def plot_conformal_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibration_residuals: np.ndarray,
    output_names: Optional[List[str]] = None,
    n_cols: int = 3,
    alphas: Optional[List[float]] = None,
) -> None:
    """Coverage plot for conformal intervals at multiple alpha levels.

    Parameters
    ----------
    y_true : (n_samples, n_outputs)
    y_pred : (n_samples, n_outputs)
    calibration_residuals : (n_cal, n_outputs), stored residuals from fitting
    output_names : list of str, optional
    n_cols : int
    alphas : list of alpha values. Default covers common levels.
    """
    if alphas is None:
        alphas = [0.5, 0.32, 0.2, 0.1, 0.05, 0.01]

    y_true = np.atleast_2d(np.asarray(y_true))
    y_pred = np.atleast_2d(np.asarray(y_pred))
    calibration_residuals = np.atleast_2d(np.asarray(calibration_residuals))

    if y_true.shape[0] == 1:
        y_true, y_pred = y_true.T, y_pred.T
    if calibration_residuals.shape[0] == 1:
        calibration_residuals = calibration_residuals.T

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    nominal = [1 - a for a in alphas]

    empirical = []
    for i in range(n_targets):
        cov = []
        for alpha in alphas:
            n_cal = calibration_residuals.shape[0]
            q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(q_level, 1.0)
            q = float(np.quantile(calibration_residuals[:, i], q_level))
            lower = y_pred[:, i] - q
            upper = y_pred[:, i] + q
            frac = np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper))
            cov.append(frac)
        empirical.append(cov)

    empirical = np.array(empirical)

    if n_targets == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(nominal, empirical[0], 'o-', label='Empirical (conformal)')
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal')
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(output_names[0])
        ax.legend()
        plt.tight_layout()
        plt.show()
        return

    n_rows = ceil(n_targets / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        axs[i].plot(nominal, empirical[i], 'o-', label='Empirical (conformal)')
        axs[i].plot([0, 1], [0, 1], 'k--', label='Ideal')
        axs[i].set_xlabel("Nominal coverage")
        axs[i].set_ylabel("Empirical coverage")
        axs[i].set_title(output_names[i])
        axs[i].legend()

    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_conformal_vs_gaussian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    calibration_residuals: np.ndarray,
    output_names: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
) -> None:
    """Side-by-side comparison: Gaussian CI vs Conformal PI coverage.

    Shows that Gaussian intervals can be miscalibrated while conformal
    intervals achieve the target coverage.

    Parameters
    ----------
    y_true : (n_samples, n_outputs)
    y_pred : (n_samples, n_outputs)
    y_std : (n_samples, n_outputs), model uncertainty estimates
    calibration_residuals : (n_cal, n_outputs)
    output_names : list of str, optional
    alphas : list of alpha levels
    """
    from scipy.stats import norm

    if alphas is None:
        alphas = [0.5, 0.32, 0.2, 0.1, 0.05, 0.01]

    y_true = np.atleast_2d(np.asarray(y_true))
    y_pred = np.atleast_2d(np.asarray(y_pred))
    y_std = np.atleast_2d(np.asarray(y_std))
    calibration_residuals = np.atleast_2d(np.asarray(calibration_residuals))

    if y_true.shape[0] == 1:
        y_true, y_pred, y_std = y_true.T, y_pred.T, y_std.T
    if calibration_residuals.shape[0] == 1:
        calibration_residuals = calibration_residuals.T

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    nominal = [1 - a for a in alphas]

    n_cols = min(n_targets, 3)
    n_rows = ceil(n_targets / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        # Gaussian coverage
        gauss_cov = []
        for alpha in alphas:
            z = norm.ppf(1 - alpha / 2)
            lower = y_pred[:, i] - z * y_std[:, i]
            upper = y_pred[:, i] + z * y_std[:, i]
            frac = np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper))
            gauss_cov.append(frac)

        # Conformal coverage
        conf_cov = []
        n_cal = calibration_residuals.shape[0]
        for alpha in alphas:
            q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
            q_level = min(q_level, 1.0)
            q = float(np.quantile(calibration_residuals[:, i], q_level))
            lower = y_pred[:, i] - q
            upper = y_pred[:, i] + q
            frac = np.mean((y_true[:, i] >= lower) & (y_true[:, i] <= upper))
            conf_cov.append(frac)

        axs[i].plot(nominal, gauss_cov, 's-', color='C0', label='Gaussian CI')
        axs[i].plot(nominal, conf_cov, 'o-', color='C1', label='Conformal PI')
        axs[i].plot([0, 1], [0, 1], 'k--', label='Ideal')
        axs[i].set_xlabel("Nominal coverage")
        axs[i].set_ylabel("Empirical coverage")
        axs[i].set_title(output_names[i])
        axs[i].legend()

    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    fig.suptitle("Gaussian CI vs Conformal PI Coverage", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_conditional_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    n_bins: int = 10,
    output_names: Optional[List[str]] = None,
    alpha: float = 0.1,
    savefig: Optional[str] = None,
) -> None:
    """Coverage as a function of y_true value.

    Conformal prediction guarantees marginal coverage; this plot reveals
    whether coverage is approximately uniform across the response range.

    Parameters
    ----------
    y_true : (n_samples,) or (n_samples, n_outputs)
    y_lower, y_upper : same shape
    n_bins : int
    output_names : list of str, optional
    alpha : float
    savefig : str, optional
    """
    from multioutreg.conformal.metrics import conditional_coverage as _cond_cov

    y_true = np.atleast_2d(np.asarray(y_true))
    y_lower = np.atleast_2d(np.asarray(y_lower))
    y_upper = np.atleast_2d(np.asarray(y_upper))
    if y_true.shape[0] == 1:
        y_true, y_lower, y_upper = y_true.T, y_lower.T, y_upper.T

    n_targets = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_targets)]

    nominal = 1 - alpha
    n_cols = min(n_targets, 3)
    n_rows = ceil(n_targets / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axs = np.array(axs).reshape(-1)

    for i in range(n_targets):
        centers, coverages = _cond_cov(
            y_true[:, i], y_lower[:, i], y_upper[:, i], n_bins
        )
        axs[i].bar(centers, coverages, width=(centers[1] - centers[0]) * 0.8,
                    alpha=0.7, label="Empirical")
        axs[i].axhline(nominal, color='r', linestyle='--',
                        label=f"Nominal ({nominal:.0%})")
        axs[i].set_xlabel("y_true (binned)")
        axs[i].set_ylabel("Coverage")
        axs[i].set_title(output_names[i])
        axs[i].set_ylim(0, 1.05)
        axs[i].legend()

    for j in range(n_targets, n_rows * n_cols):
        axs[j].set_visible(False)

    fig.suptitle("Conditional Coverage", fontsize=16)
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
        plt.close()
    else:
        plt.show()
