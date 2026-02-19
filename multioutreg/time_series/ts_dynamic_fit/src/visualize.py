import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from src.ForecastAutoIntervals import ForecastAutoIntervals
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from typing import Any, Optional, Union

def get_predictions(
    model: Any,
    df: pd.DataFrame,
    target_col: str,
    forecast_steps: int = 0
) -> pd.Series:
    """
    Attempts to obtain predictions from the given model for the input dataframe.
    Extend this with model-specific logic as needed.

    Args:
        model: The forecasting model instance.
        df (pd.DataFrame): The input data.
        target_col (str): The name of the target column.
        forecast_steps (int): Number of forecast steps for the model, if used.

    Returns:
        pd.Series: Predicted values indexed like df.
    """
    if hasattr(model, 'predict'):
        preds = model.predict(start=0, end=len(df)-1)
    elif hasattr(model, 'forecast'):
        preds = model.forecast(steps=len(df))
    elif hasattr(model, 'predict_on_batch'):
        preds = model.predict_on_batch(df.drop(columns=[target_col]).values)
    elif hasattr(model, 'predict'):
        preds = model.predict(df.drop(columns=[target_col]).values)
    elif isinstance(model, nn.Module):
        preds = df['lstm_predictions']
    else:
        raise ValueError("Model type not supported for prediction extraction.")
    if isinstance(preds, (pd.Series, pd.DataFrame)):
        preds = preds.values.flatten()
    return pd.Series(preds, index=df.index)

def plot_predictions_vs_actual(
    df: pd.DataFrame,
    preds: Union[np.ndarray, pd.Series],
    target_col: str
) -> None:
    """
    Plot actual vs predicted values.

    Args:
        df (pd.DataFrame): The input data.
        preds (np.ndarray or pd.Series): Predicted values to plot.
        target_col (str): Target variable column name.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target_col], label='Actual')
    plt.plot(df.index, preds, label='Predicted')
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.show()

def plot_residuals(
    df: pd.DataFrame,
    preds: Union[np.ndarray, pd.Series],
    target_col: str
) -> pd.Series:
    """
    Plot residuals over time and as a histogram.

    Args:
        df (pd.DataFrame): Input data.
        preds (np.ndarray or pd.Series): Predictions from the model.
        target_col (str): Name of the target column.

    Returns:
        pd.Series: Residuals (actual - predicted).
    """
    residuals = df[target_col] - preds
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    # First subplot: Residuals over time
    axes[0].plot(df.index, residuals)
    axes[0].set_title("Residuals Over Time")
    # Second subplot: Residuals histogram
    axes[1].hist(residuals, bins=30)
    axes[1].set_title("Residuals Histogram")
    plt.tight_layout()
    plt.show()
    return residuals

def plot_acf_pacf_generic(
    residuals: Union[np.ndarray, pd.Series],
    lags: int = 40
) -> None:
    """
    Plots ACF/PACF for residuals (works on any model).

    Args:
        residuals (np.ndarray or pd.Series): Residuals from the model.
        lags (int): Number of lags to display.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(residuals, lags=lags, ax=axes[0])
    plot_pacf(residuals, lags=lags, ax=axes[1])
    axes[0].set_title("Residuals ACF")
    axes[1].set_title("Residuals PACF")
    plt.tight_layout()
    plt.show()

def plot_model_specific_diagnostics(
    model: Any
) -> None:
    """
    If available, shows model native diagnostic plots (e.g., ARIMA .plot_diagnostics()).

    Args:
        model: Model instance, possibly statsmodels.

    Returns:
        None
    """
    if hasattr(model, 'plot_diagnostics'):
        model.plot_diagnostics(figsize=(12, 8))
        plt.show()

def visualize_model(
    model: Any,
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    forecast_steps: int = 12,
    lags: int = 40
) -> None:
    """
    Main entry point: visualize model results for any time series model.

    Args:
        model: Time series forecasting model.
        df (pd.DataFrame): Input data including target and features.
        target_col (str, optional): The target variable column name. Defaults to first column in df.
        forecast_steps (int): Number of future steps to forecast and plot.
        lags (int): Number of lags for residual ACF/PACF.

    Returns:
        None
    """
    if target_col is None:
        target_col = df.columns[0]
    if isinstance(df, pd.Series):
        df = df.reset_index(drop=True).to_frame()
    # 1. Predict
    preds = get_predictions(model, df, target_col)
    
    # 2. Compare predictions
    plot_predictions_vs_actual(df, preds, target_col)
    
    # 3. Residual analysis
    residuals = plot_residuals(df, preds, target_col)
    
    # 4. ACF/PACF
    # plot_acf_pacf_generic(residuals, lags=lags)
    
    # 5. Model-specific (ARIMA, SARIMA, ETS)
    plot_model_specific_diagnostics(model)
    
    # 6. Display forecasted data, with confidence intervals
    forecaster = ForecastAutoIntervals(model)
    forecast_df = forecaster.forecast(steps=forecast_steps)
    forecaster.plot_forecast(forecast_df, history=df)
