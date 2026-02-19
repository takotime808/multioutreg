from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ForecastAutoIntervals:
    def __init__(self, model: Any) -> None:
        """
        Initializes the ForecastAutoIntervals class.

        Args:
            model (Any): The forecasting model instance.
        """
        self.model = model

    def forecast(
        self,
        steps: int = 1,
        alpha: float = 0.05,
        X: Optional[Any] = None,
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Generates forecast with confidence intervals using the provided model.

        Handles statsmodels-style, quantile regressors, and generic models with bootstrap intervals.

        Args:
            steps (int): Number of steps to forecast.
            alpha (float): Significance level for confidence intervals.
            X (Any, optional): Exogenous variables for the forecast.
            y_train (pd.Series or np.ndarray, optional): Training target values for fallback.

        Returns:
            pd.DataFrame: DataFrame containing columns 'mean', 'lower', 'upper'.

        Raises:
            NotImplementedError: If the model type is unsupported or insufficient data is provided.
        """
        # Statsmodels-style models (ARIMA, etc)
        if hasattr(self.model, 'get_forecast'):
            pred = self.model.get_forecast(steps=steps)
            mean = pred.predicted_mean
            conf_int = pred.conf_int(alpha=alpha)
            return pd.DataFrame({
                'mean': mean,
                'lower': conf_int.iloc[:, 0],
                'upper': conf_int.iloc[:, 1]
            })
        # Quantile Regressors
        elif (
            hasattr(self.model, 'predict') and 
            hasattr(self.model, 'set_params') and
            hasattr(self.model, 'quantile')
        ):
            preds = {}
            quantiles = [alpha/2, 1-alpha/2]
            preds['mean'] = self.model.predict(X)
            originals = self.model.quantile
            for q, label in zip(quantiles, ['lower', 'upper']):
                self.model.set_params(quantile=q)
                preds[label] = self.model.predict(X)
            # Restore original quantile
            self.model.set_params(quantile=originals)
            return pd.DataFrame(preds)
        # Generic, fallback: bootstrap intervals
        elif hasattr(self.model, 'predict') and y_train is not None and X is not None:
            preds = self.model.predict(X)
            # Basic bootstrap from residuals
            residuals = y_train - self.model.predict(X[:len(y_train)])
            boot_preds = [preds + np.random.choice(residuals, size=len(preds), replace=True)
                          for _ in range(1000)]
            lower = np.percentile(boot_preds, 100 * alpha/2, axis=0)
            upper = np.percentile(boot_preds, 100 * (1-alpha/2), axis=0)
            return pd.DataFrame({'mean': preds, 'lower': lower, 'upper': upper})
        else:
            raise NotImplementedError("Model type not recognized or not enough data for fallback.")

    def plot_forecast(
        self,
        forecast_df: pd.DataFrame,
        history: Optional[Union[pd.Series, pd.DataFrame]] = None,
        history_label: str = 'History',
        forecast_label: str = 'Forecast'
    ) -> None:
        """
        Plots the forecast and its confidence interval on an integer-based x-axis.
    
        This method displays both historical values and forecasted values along 
        a continuous x-axis of sequential integers. Integer values are used for 
        the x-axis instead of datetime indices, ensuring proper alignment and 
        eliminating issues caused by irregular date indices or frequency mismatches.
    
        The most recent historical value is placed at position 0. All preceding 
        historical values appear at negative integer steps (e.g., -n, ..., -2, -1).
        Forecasted values start from 0 and increment positively (0, 1, ..., steps-1).
    
        Args:
            forecast_df (pd.DataFrame): 
                DataFrame containing forecasted results with columns ['mean', 'lower', 'upper'].
            history (pd.Series or pd.DataFrame, optional): 
                Historical time series data preceding the forecast. 
                If a DataFrame with one column, will be converted to a Series automatically.
            history_label (str): 
                Legend label for the plotted historical data.
            forecast_label (str): 
                Legend label for the plotted forecast.
    
        Returns:
            None
        """
        plt.figure(figsize=(12, 6))
    
        n_history = len(history) if history is not None else 0
        n_forecast = len(forecast_df)
    
        # Integer x-axis mapping
        x_history = np.arange(-n_history, 0)
        x_forecast = np.arange(0, n_forecast)
    
        # Plot history using integers
        if history is not None:
            # Convert to 1D array if DataFrame with one column
            if isinstance(history, pd.DataFrame) and history.shape[1] == 1:
                y_history = history.iloc[:, 0].values
            else:
                y_history = np.asarray(history)
            sns.lineplot(x=x_history, y=y_history, label=history_label, color='blue')
    
        # Forecast mean, lower, upper values
        y_forecast = forecast_df['mean'].values
        y_lower = forecast_df['lower'].values
        y_upper = forecast_df['upper'].values
    
        # Plot forecast using integers
        sns.lineplot(x=x_forecast, y=y_forecast, label=forecast_label, color='orange')
    
        # Confidence intervals
        plt.fill_between(x_forecast, y_lower, y_upper, color='orange', alpha=0.3, label='Confidence Interval')
    
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Forecast with Confidence Interval')
        plt.tight_layout()
        plt.show()