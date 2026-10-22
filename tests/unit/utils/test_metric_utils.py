# Copyright (c) 2025 takotime808

import numpy as np
import pytest
import pandas as pd

from multioutreg.utils.metric_utils import flatten_metrics_df, gaussian_uncertainty_metrics


def test_flatten_metrics_df():
    data = {
        'accuracy': [{'mae': 1.0, 'rmse': 2.0}],
        'sharpness': [{'var': 0.1}],
        'output': [0]
    }
    df = pd.DataFrame(data)
    flat = flatten_metrics_df(df)
    assert set(flat.columns) == {'mae', 'rmse', 'var', 'output'}
    assert flat.loc[0, 'mae'] == 1.0
    assert flat.loc[0, 'var'] == 0.1


def test_gaussian_uncertainty_metrics():
    y = np.zeros((4, 2))
    pred = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    std = np.ones((4, 2))
    m = gaussian_uncertainty_metrics(y, pred, std)
    assert m["Empirical coverage of 95% Gaussian interval"] == pytest.approx(7 / 8)
    assert m["Miscoverage at 95% (|nominal − empirical|)"] == pytest.approx(0.075)
    assert m["Sharpness (mean predicted std)"] == pytest.approx(1.0)
    assert "dummy_metric" not in m
    assert m["Gaussian NLL (mean per point)"] > 0


def test_gaussian_uncertainty_metrics_zero_std_omits_nll():
    y = np.zeros((3, 1))
    m = gaussian_uncertainty_metrics(y, y, np.zeros((3, 1)), nominal=0.9)
    assert "Gaussian NLL (mean per point)" not in m
    assert m["Empirical coverage of 90% Gaussian interval"] == pytest.approx(1.0)
