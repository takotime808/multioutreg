# Copyright (c) 2025 takotime808

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# Dynamically load the numbered module
FILE_PATH = os.path.abspath("multioutreg/gui/pages/03_Sample_Bias_Alert.py")
spec = importlib.util.spec_from_file_location("sample_bias_alert", FILE_PATH)
sba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sba)


@pytest.fixture
def simple_dataframe():
    data = {
        "A": [1, 2, 3, 4, 5, 1],
        "B": [5, 4, 3, 2, 1, 5],
        "C": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # zero variance
    }
    return pd.DataFrame(data)


def test_compute_bias_metrics(simple_dataframe):
    metrics = sba.compute_bias_metrics(simple_dataframe)
    assert isinstance(metrics, dict)
    assert "duplicate_rate" in metrics
    assert "max_abs_corr" in metrics
    assert "min_variance" in metrics
    assert "max_variance" in metrics
    assert metrics["duplicate_rate"] == pytest.approx(1/6)
    assert metrics["min_variance"] == 0.0
    assert metrics["max_variance"] > 0.0
    assert metrics["max_abs_corr"] <= 1.0


def test_compute_bias_metrics_with_no_numeric_columns():
    df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["x", "y", "z"]})
    metrics = sba.compute_bias_metrics(df)
    assert metrics["duplicate_rate"] == 0.0
    assert metrics["max_abs_corr"] == 0.0
    assert metrics["min_variance"] == 0.0
    assert metrics["max_variance"] == 0.0


def test_scan_logs_with_cholesky_warning():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("Matrix is not positive definite. Cholesky failed.")
        f_path = f.name
    alerts = sba.scan_logs(f_path)
    assert any("cholesky" in alert.lower() for alert in alerts)
    os.remove(f_path)


def test_scan_logs_without_warning():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("All good.")
        f_path = f.name
    alerts = sba.scan_logs(f_path)
    assert alerts == []
    os.remove(f_path)


def test_scan_logs_invalid_path():
    alerts = sba.scan_logs("non_existent_path.log")
    assert alerts == []


def test_pca_plot_returns_figure(simple_dataframe):
    fig = sba.pca_plot(simple_dataframe)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_title() == "PCA Projection"
    assert "PC1" in ax.get_xlabel()
    assert "PC2" in ax.get_ylabel()
