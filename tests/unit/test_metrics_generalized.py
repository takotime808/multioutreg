# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from multioutreg.performance.metrics_generalized_api import get_uq_performance_metrics_flexible


def test_get_uq_performance_metrics_flexible():
    rng = np.random.RandomState(1)
    X = rng.rand(40, 4)
    Y = np.dot(X, rng.rand(4, 3)) + rng.randn(40, 3) * 0.05
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    model = MultiOutputRegressor(GaussianProcessRegressor())
    model.fit(X_train, y_train)

    metrics_df, overall = get_uq_performance_metrics_flexible(model, X_test, y_test)
    assert not metrics_df.empty
    assert {'rmse', 'mae', 'nll', 'miscal_area', 'output'} <= set(metrics_df.columns)
    assert 'mae' in overall