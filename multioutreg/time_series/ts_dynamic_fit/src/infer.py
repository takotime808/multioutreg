# Copyright (c) 2025 takotime808
import json
from multioutreg.time_series.ts_dynamic_fit.algs.arima import ARIMA
from multioutreg.time_series.ts_dynamic_fit.algs.sarima import SARIMA
from multioutreg.time_series.ts_dynamic_fit.algs.lstm import LSTM, LSTMModel
import pickle
from typing import Sequence, Any
import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def prediction_interval(future_preds: np.ndarray, rmsfe: float) -> None:
    """
    Prints a 95% confidence prediction interval for each forecasted value in the sequence.

    Args:
        future_preds (np.ndarray): Array of forecasted values.
        rmsfe (float): Root Mean Squared Forecast Error for interval calculation.

    Returns:
        None

    Raises:
        None
    """
    z = 1.96  # 95% confidence
    
    horizons = np.arange(1, len(future_preds)+1)
    stds = rmsfe * np.sqrt(horizons)
    upper_bounds = future_preds + z * stds
    lower_bounds = future_preds - z * stds

    print("Predictions with 95% step associated decaying prediction interval:\n")
    for i, (pred, lower, upper) in enumerate(zip(future_preds, lower_bounds, upper_bounds), 1):
        print(f"Step {i}: {pred:.2f} (95% PI: [{lower:.2f}, {upper:.2f}])")

def infer(
    processed_data: pd.DataFrame, 
    feature_column: str, 
    metadata_path: str = "Model Registry/example/metadata.json", 
    model_path: str = "Model Registry/example/LSTM.pt"
) -> np.ndarray:
    """
    Runs inference using the specified model (ARIMA, SARIMA, or LSTM) and returns future predictions.

    Args:
        processed_data (pd.DataFrame): Input data for inference.
        feature_column (str): Column name containing values for prediction.
        metadata_path (str, optional): Path to metadata JSON file. Defaults to 'Model Registry/example/metadata.json'.
        model_path (str, optional): Path to serialized model file. Defaults to 'Model Registry/example/LSTM.pt'.

    Returns:
        np.ndarray: Array of future predicted values.

    Raises:
        ValueError: If initialization values for differencing are missing.
    """
    with open(metadata_path, 'r') as file:
        data = json.load(file)
        try:
            diff_count = data['metadata']['diff_count']
        except Exception:
            pass
        init_vals = data['K-order data']
    
    if data['performance']["Model"] in {"ARIMA", "SARIMA"}:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        last_init_vals = []
        for i in range(diff_count):
            diff_key = f'diff_{i}_data'
            if isinstance(init_vals.get(diff_key), dict) and init_vals[diff_key]:
                max_idx = max(init_vals[diff_key], key=lambda x: int(x))
                last_val = init_vals[diff_key][max_idx]
                last_init_vals.append(last_val)
            else:
                raise ValueError(f"Missing initialization value for {diff_key}")
        
        predictions = model.forecast(steps=5)
        future_preds = np.array(predictions)
        for k in reversed(range(diff_count)):
            future_preds = np.concatenate(([last_init_vals[k]], future_preds)).cumsum()
            future_preds = future_preds[1:]
    
    elif data['performance']["Model"] == "LSTM":
        model = LSTMModel(
            input_size=data['metadata']['input_size'], 
            hidden_layer_size=data['metadata']['hidden_layer_size'],
            num_layers=data['metadata']['num_layers'], 
            output_size=data['metadata']['output_size']
        )
        model.load_state_dict(torch.load(model_path))
        scaler = joblib.load("temp/scaler.pkl")
        train_window = 10  
        test_inputs = processed_data[feature_column][-train_window:].values
        test_inputs_scaled = scaler.transform(test_inputs.reshape(-1, 1)).flatten()
        seq = torch.FloatTensor(test_inputs_scaled)
        with torch.no_grad():
            preds = model(seq)
        preds_np = preds.detach().numpy().reshape(-1, 1)
        future_preds = scaler.inverse_transform(preds_np).flatten()
        future_dates = pd.date_range(processed_data[feature_column].index[-1] + pd.Timedelta(days=1), periods=len(future_preds))

    print(f"The next 10 predictions are:\n {future_preds}\n")
    prediction_interval(future_preds, data['performance']['RMSE'])

    return future_preds

def plot_inference(processed_data: pd.DataFrame, future_preds: np.ndarray) -> None:
    """
    Plots actual data vs forecasted future predictions on a date axis.

    Args:
        processed_data (pd.DataFrame): The original processed time series data.
        future_preds (np.ndarray): The predicted future values array.

    Returns:
        None

    Raises:
        None
    """
    future_dates = pd.date_range(processed_data.index[-1] + pd.Timedelta(days=1), periods=len(future_preds))
    
    combined_df = pd.concat([
        processed_data,
        pd.Series(future_preds, index=future_dates)
    ])
    
    plt.figure(figsize=(10, 5))
    plt.plot(processed_data.index, processed_data.values, label="Actual")
    plt.plot(future_dates, future_preds, label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Actual vs Forecasted Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

