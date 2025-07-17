# Copyright (c) 2025 takotime808

import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def filter_subset(arrays, n_subset):
    idx = np.random.choice(len(arrays[0]), size=n_subset, replace=False)
    return [a[idx] for a in arrays]


def get_proportion_lists_vectorized(y_pred, y_std, y_true, prop_type="interval"):
    # Use 100 intervals from 0 to 1
    exp_props = np.linspace(0, 1, 100)
    obs_props = []
    for alpha in exp_props:
        z = abs(np.percentile(np.random.normal(0,1,10000), 100 * (1-alpha/2)))
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        prop = np.mean((y_true >= lower) & (y_true <= upper))
        obs_props.append(prop)
    return exp_props, np.array(obs_props)


def miscalibration_area_from_proportions(exp_proportions, obs_proportions):
    # Area between curve and diagonal line (trapezoid rule)
    return np.trapz(np.abs(exp_proportions - obs_proportions), exp_proportions)


def plot_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    curve_label: Union[str, None] = None,
    vectorized: bool = True,
    exp_props: Union[np.ndarray, None] = None,
    obs_props: Union[np.ndarray, None] = None,
    ax: Union[plt.Axes, None] = None,
    prop_type: str = "interval",
) -> plt.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        if vectorized:
            exp_proportions, obs_proportions = get_proportion_lists_vectorized(
                y_pred, y_std, y_true, prop_type=prop_type
            )
        else:
            raise NotImplementedError("Only vectorized supported in this demo.")
    else:
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    if curve_label is None:
        curve_label = "Predictor"

    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(exp_proportions, obs_proportions, label=curve_label, c="#1f77b4")
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)

    ax.set_xlabel("Predicted Proportion in Interval")
    ax.set_ylabel("Observed Proportion in Interval")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration")

    miscalibration_area = miscalibration_area_from_proportions(
        exp_proportions=exp_proportions, obs_proportions=obs_proportions
    )

    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )
    return ax

