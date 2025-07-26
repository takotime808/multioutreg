# Copyright (c) 2025 takotime808

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from multioutreg.figures.pdp_plots import generate_pdp_plot


@pytest.fixture
def dummy_data():
    X = np.random.rand(10, 3)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    output_names = [f"output_{i}" for i in range(2)]
    return X, feature_names, output_names


@pytest.fixture
def mock_model():
    est1 = MagicMock()
    est2 = MagicMock()
    model = MagicMock()
    model.estimators_ = [est1, est2]
    return model


@patch("multioutreg.figures.pdp_plots.plot_to_b64")
@patch("multioutreg.figures.pdp_plots.PartialDependenceDisplay.from_estimator")
def test_generate_pdp_plot_success(mock_pdp_display, mock_plot_to_b64, mock_model, dummy_data):
    X, feature_names, output_names = dummy_data

    # Force execution of plot_fn inside plot_to_b64
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64image==")

    results = generate_pdp_plot(mock_model, X, output_names, feature_names)

    assert isinstance(results, dict)
    assert set(results.keys()) == set(output_names)
    assert all(v == "base64image==" for v in results.values())
    assert mock_pdp_display.call_count == 2


@patch("multioutreg.figures.pdp_plots.plot_to_b64")
@patch("multioutreg.figures.pdp_plots.PartialDependenceDisplay.from_estimator")
def test_generate_pdp_plot_with_failure(mock_pdp_display, mock_plot_to_b64, mock_model, dummy_data):
    X, feature_names, output_names = dummy_data

    def side_effect(estimator, X_, features, **kwargs):
        if estimator == mock_model.estimators_[0]:
            raise ValueError("Unsupported model")
        return MagicMock()

    mock_pdp_display.side_effect = side_effect
    mock_plot_to_b64.side_effect = lambda fn: (fn() or "base64img")

    result = generate_pdp_plot(mock_model, X, output_names, feature_names)

    assert result.keys() == set(output_names)
    assert all(val == "base64img" for val in result.values())
    assert mock_pdp_display.call_count == 2
    assert mock_plot_to_b64.call_count == 2
