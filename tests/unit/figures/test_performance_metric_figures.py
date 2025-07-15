# Copyright (c) 2025 takotime808

import pytest
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multioutreg.figures.performance_metric_figures import plot_uq_metrics_bar

def make_flat_df():
    return pd.DataFrame({
        "output": [0, 1],
        "rmse": [0.2, 0.3],
        "mae": [0.15, 0.25],
        "nll": [1.0, 0.8],
        "miscal_area": [0.1, 0.2]
    })

def make_nested_df():
    return pd.DataFrame({
        "output": [0, 1],
        "accuracy": [{"rmse": 0.2, "mae": 0.15}, {"rmse": 0.3, "mae": 0.25}],
        "avg_calibration": [{"nll": 1.0}, {"nll": 0.8}],
        "scoring_rule": [{"miscal_area": 0.1}, {"miscal_area": 0.2}],
    })

def test_plot_bar_flat_default_metrics():
    df = make_flat_df()
    ax = plot_uq_metrics_bar(df)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_bar_flat_subset_metrics():
    df = make_flat_df()
    ax = plot_uq_metrics_bar(df, metrics=["mae", "nll"])
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_bar_flat_given_axes():
    df = make_flat_df()
    fig, ax = plt.subplots()
    result_ax = plot_uq_metrics_bar(df, ax=ax)
    assert result_ax is ax
    plt.close(fig)

def test_plot_bar_flat_invalid_metric():
    df = make_flat_df()
    with pytest.raises(ValueError):
        plot_uq_metrics_bar(df, metrics=["bogus"])

def test_plot_bar_nested_default_metrics():
    df = make_nested_df()
    ax = plot_uq_metrics_bar(df)
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_bar_nested_subset_metrics():
    df = make_nested_df()
    ax = plot_uq_metrics_bar(df, metrics=["rmse"])
    assert isinstance(ax, plt.Axes)
    plt.close()

def test_plot_bar_nested_invalid_metric():
    df = make_nested_df()
    with pytest.raises(ValueError):
        plot_uq_metrics_bar(df, metrics=["bogus"])

def test_plot_bar_flat_metrics_not_present():
    df = make_flat_df()
    # All metrics requested do not exist
    with pytest.raises(ValueError):
        plot_uq_metrics_bar(df, metrics=["does_not_exist1", "does_not_exist2"])

def test_plot_bar_flat_custom_labels():
    df = make_flat_df()
    df["output"] = [3, 5]
    ax = plot_uq_metrics_bar(df)
    assert isinstance(ax, plt.Axes)
    plt.close()
