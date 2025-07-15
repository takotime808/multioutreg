# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from sklearn.model_selection import train_test_split

from multioutreg.performance import metrics as metrics_mod
from multioutreg.performance.metrics import get_uq_performance_metrics


class DummyModel:
    def predict(self, X, return_std=False):
        y_pred = np.zeros((len(X), 2))
        # use positive standard deviations to satisfy uncertainty-toolbox
        y_std = np.abs(np.random.rand(len(X), 2)) + 0.1
        if return_std:
            return y_pred, y_std
        return y_pred


def test_get_uq_performance_metrics(monkeypatch):
    rng = np.random.RandomState(0)
    X = rng.rand(50, 3)
    Y = np.dot(X, rng.rand(3, 2)) + rng.randn(50, 2) * 0.1
    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    model = DummyModel()

    def fake_get_all_metrics(*args, **kwargs):
        return {
            'accuracy': {'rmse': 0.0, 'mae': 0.0},
            'avg_calibration': {'nll': 0.0, 'miscal_area': 0.0},
            'scoring_rule': {'nll': 0.0},
        }

    monkeypatch.setattr(metrics_mod.uct, 'get_all_metrics', fake_get_all_metrics)

    metrics_df, overall = get_uq_performance_metrics(model, X_test, y_test)
    assert not metrics_df.empty
    assert {'accuracy', 'avg_calibration', 'scoring_rule', 'output'} <= set(metrics_df.columns)
    assert 'accuracy' in overall