# Copyright (c) 2026 takotime808

"""Evaluation metrics for conformal prediction intervals."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def conformal_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute empirical coverage per output.

    Parameters
    ----------
    y_true : (n_samples,) or (n_samples, n_outputs)
    y_lower : same shape as y_true
    y_upper : same shape as y_true
    output_names : list of str, optional

    Returns
    -------
    pd.DataFrame with columns: output, coverage, n_samples
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_lower = np.atleast_2d(np.asarray(y_lower))
    y_upper = np.atleast_2d(np.asarray(y_upper))

    if y_true.shape[0] == 1:
        y_true, y_lower, y_upper = y_true.T, y_lower.T, y_upper.T

    n_outputs = y_true.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_outputs)]

    rows = []
    for j in range(n_outputs):
        covered = (y_true[:, j] >= y_lower[:, j]) & (y_true[:, j] <= y_upper[:, j])
        rows.append({
            "output": output_names[j],
            "coverage": float(np.mean(covered)),
            "n_samples": len(covered),
        })
    return pd.DataFrame(rows)


def conformal_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    output_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute interval width statistics per output.

    Returns
    -------
    pd.DataFrame with columns: output, mean_width, median_width, std_width
    """
    y_lower = np.atleast_2d(np.asarray(y_lower))
    y_upper = np.atleast_2d(np.asarray(y_upper))

    if y_lower.shape[0] == 1:
        y_lower, y_upper = y_lower.T, y_upper.T

    n_outputs = y_lower.shape[1]
    if output_names is None:
        output_names = [f"Output {i}" for i in range(n_outputs)]

    rows = []
    for j in range(n_outputs):
        widths = y_upper[:, j] - y_lower[:, j]
        rows.append({
            "output": output_names[j],
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "std_width": float(np.std(widths)),
        })
    return pd.DataFrame(rows)


def conformal_summary(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float,
    output_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Combined coverage and width summary.

    Returns
    -------
    pd.DataFrame with columns: output, nominal_coverage, empirical_coverage,
    mean_width, median_width, coverage_gap
    """
    cov_df = conformal_coverage(y_true, y_lower, y_upper, output_names)
    width_df = conformal_interval_width(y_lower, y_upper, output_names)

    summary = cov_df.merge(width_df, on="output")
    summary["nominal_coverage"] = 1 - alpha
    summary["coverage_gap"] = summary["coverage"] - summary["nominal_coverage"]
    return summary[
        ["output", "nominal_coverage", "coverage", "coverage_gap",
         "mean_width", "median_width", "n_samples"]
    ]


def conditional_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assess conditional coverage by binning y_true.

    For single-output only (1D arrays).

    Parameters
    ----------
    y_true : (n_samples,)
    y_lower : (n_samples,)
    y_upper : (n_samples,)
    n_bins : int

    Returns
    -------
    bin_centers : (n_bins,)
    bin_coverages : (n_bins,)
    """
    y_true = np.asarray(y_true).ravel()
    y_lower = np.asarray(y_lower).ravel()
    y_upper = np.asarray(y_upper).ravel()

    covered = (y_true >= y_lower) & (y_true <= y_upper)

    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_coverages = np.zeros(n_bins)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        else:
            mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_coverages[i] = covered[mask].mean()
        else:
            bin_coverages[i] = np.nan

    return bin_centers, bin_coverages
