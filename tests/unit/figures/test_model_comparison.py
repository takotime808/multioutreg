# Copyright (c) 2025 takotime808

import os
import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from multioutreg.figures.model_comparison import plot_surrogate_model_summary


@pytest.fixture
def regression_data(request):
    n_targets = request.param
    X, Y = make_regression(n_samples=100, n_features=5, n_targets=n_targets, noise=0.5, random_state=42)
    return train_test_split(X, Y, test_size=0.2, random_state=1)


@pytest.mark.parametrize("regression_data", [2, 3], indirect=True)
def test_returns_figure_if_no_save(regression_data):
    X_train, X_test, Y_train, Y_test = regression_data
    fig = plot_surrogate_model_summary(X_train, X_test, Y_train, Y_test, savefig=None, admin_mode=False)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.parametrize("regression_data", [2, 3], indirect=True)
def test_saves_file_if_savefig_given(regression_data, tmp_path):
    X_train, X_test, Y_train, Y_test = regression_data
    save_path = tmp_path / "test_output.png"
    result = plot_surrogate_model_summary(X_train, X_test, Y_train, Y_test, savefig=str(save_path), admin_mode=False)
    assert isinstance(result, str)
    assert os.path.exists(result)
    assert result.endswith(".png")


@pytest.mark.parametrize("regression_data", [3], indirect=True)
def test_rmse_plot_index_positioning(regression_data):
    X_train, X_test, Y_train, Y_test = regression_data
    # Valid plot index
    try:
        plot_surrogate_model_summary(X_train, X_test, Y_train, Y_test, rmse_plot_index=2, savefig=None, admin_mode=False)
    except Exception as e:
        pytest.fail(f"Valid rmse_plot_index raised an error: {e}")

    # Out-of-bounds plot index
    with pytest.raises(IndexError):
        plot_surrogate_model_summary(X_train, X_test, Y_train, Y_test, rmse_plot_index=5, savefig=None, admin_mode=False)


@pytest.mark.parametrize("regression_data", [2], indirect=True)
def test_compare_false_removes_noise_model(regression_data):
    X_train, X_test, Y_train, Y_test = regression_data
    # Should run without error and no noise model comparison
    result = plot_surrogate_model_summary(X_train, X_test, Y_train, Y_test, compare=False, savefig=None, admin_mode=False)
    assert result is not None

def test_custom_model_override():
    # Use a single-target dataset
    X, Y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y.reshape(-1, 1), test_size=0.3, random_state=0)
    
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train, Y_train)

    # rmse_plot_index must be 0 when n_targets == 1
    result = plot_surrogate_model_summary(
        X_train, X_test, Y_train, Y_test,
        model=model,
        rmse_plot_index=0,  # ✅ fix
        savefig=None,
        admin_mode=False
    )
    assert result is not None
