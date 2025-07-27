# Copyright (c) 2025 takotime808

import numpy as np
import pytest
from sklearn.datasets import make_regression
import matplotlib
matplotlib.use('Agg')  # for headless test environments
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from multioutreg.gui.pages.Use_Auto_Detect_MultiOutput_Regressor import (
    RandomForestWithUncertainty,
    GradientBoostingWithUncertainty,
    KNeighborsRegressorWithUncertainty,
    BootstrapLinearRegression,
    PerTargetRegressorWithStd,
    generate_html_report,
)
# from sklearn.ensemble import RandomForestRegressor
from multioutreg.model_selection import AutoDetectMultiOutputRegressor

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=5, n_targets=2, noise=0.1, random_state=42)
    return X, y


def test_random_forest_with_uncertainty_predict_shape(regression_data):
    X, y = regression_data
    model = RandomForestWithUncertainty(n_estimators=10)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape


def test_gradient_boosting_with_uncertainty_predict_shape(regression_data):
    X, y = regression_data
    # GradientBoostingWithUncertainty only supports 1D y
    y_single = y[:, 0]
    model = GradientBoostingWithUncertainty(n_estimators=10)
    model.fit(X, y_single)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == (X.shape[0],)
    assert std.shape == (X.shape[0],)


def test_knn_with_uncertainty_predict_shape(regression_data):
    X, y = regression_data
    model = KNeighborsRegressorWithUncertainty(n_neighbors=5)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape


def test_bootstrap_linear_regression_predict_shape(regression_data):
    X, y = regression_data
    model = BootstrapLinearRegression(n_bootstraps=5)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape


def test_per_target_regressor_with_std_predict_shape(regression_data):
    X, y = regression_data
    from sklearn.linear_model import LinearRegression
    base_models = [LinearRegression() for _ in range(y.shape[1])]
    model = PerTargetRegressorWithStd(base_models)
    model.fit(X, y)
    mean, std = model.predict(X, return_std=True)
    assert mean.shape == y.shape
    assert std.shape == y.shape


def test_generate_html_report_returns_str(tmp_path):
    X, y = make_regression(n_samples=50, n_features=3, n_targets=2, random_state=0)
    y_pred = y + np.random.normal(0, 0.1, y.shape)
    y_std = np.full_like(y, 0.05)
    feature_names = ["x1", "x2", "x3"]

    base_model = RandomForestWithUncertainty(n_estimators=5)
    model = AutoDetectMultiOutputRegressor([base_model, base_model], [{}, {}])
    model.fit(X, y)

    report = generate_html_report(
        model_type="TestModel",
        fidelity_levels=[],
        output_names=["y1", "y2"],
        description="Test description",
        metrics={"y1": {"r2": 1.0, "rmse": 0.1, "mae": 0.1, "mean_predicted_std": 0.05},
                 "y2": {"r2": 1.0, "rmse": 0.1, "mae": 0.1, "mean_predicted_std": 0.05}},
        uncertainty_metrics={"dummy_metric": 0.1},
        y_test=y,
        best_pred=y_pred,
        best_std=y_std,
        best_model=model,
        X_train=X,
        n_train=40,
        n_test=10,
        cross_validation="None",
        seed=42,
        notes="Test run",
        feature_names=feature_names,  # ✅ Add this line
        feature_names_pca=["PC1", "PC2", "PC3"],
        pca_explained_variance=[0.8, 0.15, 0.05],
        pca_variance_plot="",
        pca_method="Manual",
        pca_threshold=None,
        pca_n_components=3,
        kaiser_rule_suggestion="Use 3 components"
    )

    assert isinstance(report, str)
    assert "Multi-Fidelity" in report


@pytest.fixture
def synthetic_multioutput_data():
    rng = np.random.RandomState(0)
    X = rng.rand(200, 4)
    y_linear = X @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.randn(200) * 0.01
    y_tree = np.sin(X[:, 0]) + rng.randn(200) * 0.01
    Y = np.column_stack([y_linear, y_tree])
    return X, Y


def test_manual_estimator_selection(synthetic_multioutput_data):
    X, Y = synthetic_multioutput_data
    estimators = [LinearRegression(), DecisionTreeRegressor(random_state=0)]
    param_spaces = [{}, {"max_depth": [1, None]}]

    model = AutoDetectMultiOutputRegressor(estimators, param_spaces)
    model.fit(X, Y)
    preds = model.predict(X)

    assert preds.shape == Y.shape
    assert len(model.models_) == Y.shape[1]
    assert all(hasattr(m, "predict") for m in model.models_)


def test_manual_predict_with_std(synthetic_multioutput_data):
    X, Y = synthetic_multioutput_data
    estimators = [LinearRegression(), DecisionTreeRegressor(random_state=0)]
    param_spaces = [{}, {"max_depth": [1, None]}]

    model = AutoDetectMultiOutputRegressor(estimators, param_spaces)
    model.fit(X, Y)
    preds, stds = model.predict(X[:10], return_std=True)

    assert preds.shape == stds.shape
    assert preds.shape == (10, Y.shape[1])
    assert np.all(stds >= 0)


def test_vendored_surrogate_search(synthetic_multioutput_data):
    X, Y = synthetic_multioutput_data
    model = AutoDetectMultiOutputRegressor.with_vendored_surrogates()
    model.fit(X, Y)
    preds, stds = model.predict(X[:5], return_std=True)

    assert preds.shape == stds.shape
    assert preds.shape == (5, Y.shape[1])
    assert all(hasattr(m, "predict") for m in model.models_)

    model_names = [type(m).__name__ for m in model.models_]
    print("Selected models:", model_names)
    print("Predictions:", preds)
    print("Standard deviations:", stds)
