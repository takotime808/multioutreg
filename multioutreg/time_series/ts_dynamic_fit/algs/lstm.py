from keras.models import Sequential
from keras.layers import Dense, LSTM as KerasLSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import logging
import joblib
from typing import Tuple, Dict, Any, Optional

class LSTMModel(nn.Module):


    def __init__(self, input_size: int = 1, hidden_layer_size: int = 128, num_layers: int = 2, output_size: int = 10):
       
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass to produce a multi-step forecast.

        Args:
            input_seq (torch.FloatTensor): Sequence data for prediction.

        Returns:
            torch.FloatTensor: Model outputs for multi-step forecast.
        """
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out[-1])
        return predictions
    
class LSTM:

    def __init__(
        self, 
        df: pd.DataFrame, 
        fut_pred: int, 
        feature_column: str = "customers", 
        train_window: int = 10, 
        verbose: bool = False, 
        log_filename: str = "pipeline.log"
    ):
        
        self.verbose = verbose
        self.data = df[feature_column]
        self.orig_data = df[feature_column]
        self.train_window = train_window
        self.log_filename = log_filename
        self.scaled_data = None

    def log_to_file(self, log: str) -> None:
        """
        Appends log message to file.

        Args:
            log (str): Log message.

        Returns:
            None
        """
        logging.info(log)
        with open(f"logs/{self.log_filename}", 'a') as f:
            f.write(log + '\n')

    def create_inout_sequences(
        self, 
        input_data: torch.FloatTensor, 
        window: int, 
        fut_pred: int
    ) -> list:
        """
        Creates input-output sequences for training.

        Args:
            input_data (torch.FloatTensor): Scaled time series data.
            window (int): Sequence window size.
            fut_pred (int): Prediction horizon.

        Returns:
            list: List of (train_seq, train_label) tuples for training.
        """
        inout_seq = []
        L = len(input_data)
        for i in range(L - window - fut_pred + 1):
            train_seq = input_data[i:i+window]
            train_label = input_data[i+window:i+window+fut_pred]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def visualize(
        self, 
        historical_df: pd.Series, 
        future_preds: np.ndarray, 
        fut_pred: int, 
        resid_std: Optional[float] = None
    ) -> None:
        """
        Plots actual values, forecasts, and optionally confidence intervals.

        Args:
            historical_df (pd.Series): Original historical time series data.
            future_preds (np.ndarray): Future predictions from the model.
            fut_pred (int): Prediction horizon.
            resid_std (Optional[float]): Residual standard deviation (for interval).

        Returns:
            None
        """
        historical_df = historical_df.copy()
        historical_df.index = pd.to_datetime(historical_df.index)
        last_date = historical_df.index[-1]
    
        df_past = historical_df.to_frame(name='Actual').reset_index()
        df_past.rename(columns={'index': 'Date'}, inplace=True)
        df_past['Forecast'] = np.nan
        df_past.loc[df_past.index[-1], 'Forecast'] = df_past['Actual'].iloc[-1]
    
        df_future = pd.DataFrame({
            'Date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=fut_pred),
            'Forecast': future_preds.flatten(),
            'Actual': np.nan
        })
        print("#"*40)
        print(future_preds.flatten())
        results = pd.concat([df_past, df_future], ignore_index=True).reset_index()
        forecast_values = results[len(df_past):].index
        if resid_std is not None:
            lower = df_future['Forecast'] - 1.96 * resid_std
            upper = df_future['Forecast'] + 1.96 * resid_std
    
        plt.figure(figsize=(12,6))
        plt.plot(results.index, results['Actual'], label='Actual')
        plt.plot(results.index, results['Forecast'], label='Forecast')
        if resid_std is not None:
            plt.fill_between(results[len(df_past):].index, lower, upper, color="orange", alpha=0.3, label="95% Confidence Interval")
        plt.title("Actual and Forecasted Values by LSTM")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def convert_keys_to_str(self, obj: Any) -> Any:
        """
        Recursively converts dictionary keys to strings if pandas timestamps.

        Args:
            obj (Any): Input dictionary or list.

        Returns:
            Any: Object with keys converted to strings.
        """
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, pd.Timestamp) else k: self.convert_keys_to_str(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_keys_to_str(item) for item in obj]
        else:
            return obj

    def run(self) -> Tuple[Dict[str, Any], pd.Series, LSTMModel]:
        """
        Trains LSTM model, predicts, evaluates, and returns metadata and objects.

        Returns:
            Tuple[Dict[str, Any], pd.Series, LSTMModel]:
                - performance and metadata dictionary
                - original time series
                - trained model

        Raises:
            None
        """
        self.log_to_file("#"*40)
        self.log_to_file("Starting LSTM Training Loop...")
        self.log_to_file("#"*40)

        scaler = MinMaxScaler(feature_range=(-1, 1))
    
        model = LSTMModel()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

        fut_pred = 10
        self.data = scaler.fit_transform(self.data.to_frame())
        self.scaled_data = self.data
        self.data = torch.FloatTensor(self.data).view(-1)
        self.orig_data.index = pd.to_datetime(self.orig_data.index)
        last_date = self.orig_data.index[-1]
    
        train_seq = self.create_inout_sequences(self.data, window=self.train_window, fut_pred=fut_pred)
    
        model = LSTMModel(output_size=fut_pred)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
        epochs = 200
        for i in range(epochs):
            for seq, labels in train_seq:
                optimizer.zero_grad()
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()
            if self.verbose:
                if i % 25 == 1:
                    self.log_to_file(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        if self.verbose:
            self.log_to_file(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    
        model.eval()
        test_inputs = self.data[-self.train_window:].tolist()
        seq = torch.FloatTensor(test_inputs)
        with torch.no_grad():
            preds = model(seq)
        preds = preds.detach().numpy().reshape(-1, 1)
        preds = scaler.inverse_transform(preds)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=fut_pred, freq='D')
        df = pd.DataFrame({'lstm_pred': preds.flatten()}, index=future_dates)

        y_trues = []
        y_preds = []
        for seq, labels in train_seq:
            with torch.no_grad():
                y_pred = model(seq)
                y_trues.append(labels.cpu().numpy())
                y_preds.append(y_pred.cpu().numpy())
        
        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)
        y_trues_flat = y_trues.reshape(-1, 1)
        y_preds_flat = y_preds.reshape(-1, 1)
        y_trues_orig = scaler.inverse_transform(y_trues_flat).flatten()
        y_preds_orig = scaler.inverse_transform(y_preds_flat).flatten()
        residuals = y_trues_orig - y_preds_orig
        resid_std = np.std(residuals)
        rmse = np.sqrt(np.mean((y_trues_orig - y_preds_orig) ** 2))
        mae_val = np.mean(np.abs(y_trues_orig - y_preds_orig))
        mask = np.abs(y_trues_orig) >= 1
        y_trues_orig_filtered = y_trues_orig[mask]
        y_preds_orig_filtered = y_preds_orig[mask]
        mape_val = np.mean(np.abs((y_trues_orig_filtered - y_preds_orig_filtered) / y_trues_orig_filtered))
        n = len(y_trues_orig)
        k = sum(p.numel() for p in model.parameters())
        mse = np.mean((y_trues_orig - y_preds_orig) ** 2)
        aic = "N/A"
        bic = "N/A"

        future_preds = preds
        if self.verbose:
            self.visualize(self.orig_data, future_preds, fut_pred, resid_std=resid_std)

        joblib.dump(scaler, "temp/scaler.pkl")

        metadata: Dict[str, Any] = {
            "performance": {
                "Model": 'LSTM',
                "AIC": aic,
                "BIC": bic,
                "RMSE": float(rmse),
                "MAE": float(mae_val),
                "MAPE": float(mape_val),
            },
            "metadata": {
                "input_size": model.lstm.input_size,
                "hidden_layer_size": model.hidden_layer_size,
                "num_layers": model.lstm.num_layers,
                "output_size": model.linear.out_features,
                "train_window": self.train_window,
            },
            "K-order data": {
                "data": self.convert_keys_to_str(self.orig_data.to_dict()),
                "diff_1_data": None,
                "diff_2_data": None,
            }
        }
        return metadata, self.orig_data, model
