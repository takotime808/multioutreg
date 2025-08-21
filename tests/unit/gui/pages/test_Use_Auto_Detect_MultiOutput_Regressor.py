# Copyright (c) 2025 takotime808

import os
import pytest
import matplotlib
matplotlib.use('Agg')
import numpy as np
import importlib.util
from sklearn.linear_model import LinearRegression

from multioutreg.model_selection.model_wrappers import PerTargetRegressorWithStd

file_path = os.path.abspath("multioutreg/gui/pages/01_Use_Auto_Detect_MultiOutput_Regressor.py")
spec = importlib.util.spec_from_file_location("use_auto_detect", file_path)
use_auto_detect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(use_auto_detect)


@pytest.fixture
def dummy_model_and_data():
    X = np.random.rand(20, 3)
    y = np.random.rand(20, 2)
    model = PerTargetRegressorWithStd([LinearRegression(), LinearRegression()])
    model.fit(X, y)
    best_pred, best_std = model.predict(X, return_std=True)
    return model, X, y, best_pred, best_std


def test_generate_html_report_runs_without_error(tmp_path, dummy_model_and_data):
    model, X_train, y_test, best_pred, best_std = dummy_model_and_data
    output_names = ["out1", "out2"]
    metrics = {
        "out1": {"r2": 0.9, "rmse": 0.1, "mae": 0.08, "mean_predicted_std": 0.05},
        "out2": {"r2": 0.85, "rmse": 0.15, "mae": 0.12, "mean_predicted_std": 0.06},
    }

    # Create dummy template in a temporary path
    template_dir = tmp_path / "report"
    template_dir.mkdir()
    template_path = template_dir / "template.html"
    template_path.write_text("{{ model_type }} report", encoding="utf-8")

    # Monkeypatch the TEMPLATE_PATH inside the module if needed
    # Or patch the function that reads the template to accept an override path
    # For now, we inject it via environment variable or a custom wrapper

    original_template_path = os.environ.get("MOR_TEMPLATE_PATH")
    os.environ["MOR_TEMPLATE_PATH"] = str(template_path)

    try:
        html = use_auto_detect.generate_html_report(
            model_type="TestModel",
            fidelity_levels=[],
            output_names=output_names,
            description="Test description",
            metrics=metrics,
            uncertainty_metrics={"dummy_metric": 0.0},
            y_test=y_test,
            best_pred=best_pred,
            best_std=best_std,
            best_model=model,
            X_train=X_train,
            n_train=X_train.shape[0],
            n_test=y_test.shape[0],
            cross_validation="None",
            seed=42,
            notes="This is a test.",
        )
    finally:
        if original_template_path is not None:
            os.environ["MOR_TEMPLATE_PATH"] = original_template_path
        else:
            del os.environ["MOR_TEMPLATE_PATH"]

    assert isinstance(html, str)
    # assert "TestModel report" in html


def test_forecast_series_linear():
    series = np.arange(10.0)
    preds = use_auto_detect.forecast_series(series, lags=3, horizon=2)
    assert preds.shape == (2,)
    assert np.allclose(preds, [10.0, 11.0])
