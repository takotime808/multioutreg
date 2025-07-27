# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from multioutreg.model_selection import AutoDetectMultiOutputRegressor


def test_auto_detect_multi_output_regressor_selects_best():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y_linear = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.randn(200) * 0.01
    y_tree = np.sin(X[:, 0]) + rng.randn(200) * 0.01
    Y = np.column_stack([y_linear, y_tree])

    estimators = [LinearRegression(), DecisionTreeRegressor(random_state=0)]
    param_spaces = [{}, {"max_depth": [1, None]}]
    model = AutoDetectMultiOutputRegressor(estimators, param_spaces)
    model.fit(X, Y)
    preds = model.predict(X)

    assert preds.shape == Y.shape
    assert isinstance(model.models_[0], LinearRegression)
    assert isinstance(model.models_[1], DecisionTreeRegressor)