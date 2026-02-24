# Copyright (c) 2025 takotime808
from statsmodels.tsa.stattools import adfuller, acf, pacf
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error as mae
import logging
import warnings
from contextlib import redirect_stdout
import sys
from typing import Optional, Tuple, Any, Dict, Sequence, Union

warnings.filterwarnings("ignore")


class ARIMA:
    def __init__(self, df: pd.DataFrame, feature_column="customers", verbose=False, log_filename="pipeline.log"):
        self.diff_count = 0
        self.stationary = False
        self.visualize = False
        self.verbose = verbose
        self.log_filename = log_filename
        self.feature_column = feature_column
        self.data = df[feature_column]
        self.orig_data = df[feature_column]
        self.diff_1_data = None
        self.diff_2_data = None
        
        # Setup logging
        log_handlers = []
        # Always print to console
        log_handlers.append(logging.StreamHandler())

    def log_to_file(self, log: str) -> None:
        logging.info(log)
        with open(f"logs/{self.log_filename}", 'a') as f:
            f.write(log + '\n')
        
    def make_stationary(self, df) -> Tuple[Any, ...]:
        """
        ADF Statistic: Measures whether the series has a unit root (non-stationarity). 
            - If this number is less than the critical values, you can reject the null hypothesis
            - If not then the data is likely not stationary
            
        Args: DataFrame

        Returns: Boolean, Tuple 
            This returns a tuple with multiple values. Here is what each means:
    
            (
             ADF Statistic, :
             P-value, 
             Number of lags used in the test, 
             Number of observations used in the regression (after lagging), 
             Critical values for the test statistic at 
                 {1%, 
                  5%,
                  10%}, 
             Maximized information criterion if autolag is used (e.g., AIC, BIC, etc.)

            )
        """
        adf = adfuller(df)
        if self.verbose: 
            self.log_to_file(f"Diff {self.diff_count}...")

        if adf[1] > 0.05:
            self.diff_count += 1
            if self.verbose: 
                self.log_to_file(f"The p-value: {adf[1]} implies the series is non-stationary and requires differencing.")
            return adf
        else:
            if self.verbose:
                self.log_to_file("Data is stationary.")
            self.stationary = True
        
            return adf

    def grid_search_arima(self, p_values=range(0,8), q_values=range(0,8)):
        """
            Perform a grid search to find the optimal ARIMA model order (p, d, q) for a given time series.
            For our use "d" will always be 0 because we are only using this on data we previously made stationary. 
        
            This function iterates through all combinations of provided AR (p) and MA (q) orders,
            fits an ARIMA model for each (p, 0, q) triplet, and selects the order with the lowest AIC score.
            It returns the optimal model order along with the fitted model instance.
        
            Args:
                X (array-like or pandas.Series): The univariate time series data to fit.
                p_values (iterable of int): Sequence of candidate AR (autoregressive) orders to try.
                q_values (iterable of int): Sequence of candidate MA (moving average) orders to try.
        
            Returns:
                tuple:
                    best_order (tuple of int): The (p, q) order with the lowest AIC value found.
                    best_model (statsmodels ARIMAResults): The fitted ARIMA model instance corresponding to best_order.
            Raises:
                Any exceptions raised by statsmodels ARIMA on invalid configurations will be caught and skipped.
        """
                
        best_aic = np.inf
        best_order = None
        best_model = None
        for p in p_values:
            for q in q_values:
                try:
                    model = sm.tsa.ARIMA(self.data, order=(p,0,q))
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, 0, q)
                        best_model = results
                except:
                    continue
        return best_order, best_model
        
    def convert_keys_to_str(self, obj):
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, pd.Timestamp) else k: self.convert_keys_to_str(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_keys_to_str(item) for item in obj]
        else:
            return obj
                
    def run(self) -> Tuple[Dict[str, Any], pd.DataFrame, Any]:
        """
        Runs the ARIMA parameter search, performs differencing for stationarity,
        finds the best ARIMA model order, evaluates model performance, and returns results.
        """
        self.log_to_file("#"*40)
        self.log_to_file("Running ARIMA Parameter Search...")
        self.log_to_file("#"*40)
        
        #Iterativly perform ADF until stationary

        while not self.stationary:
            self.make_stationary(self.data)
            if not self.stationary:
                self.data = self.data.diff().dropna()

            # Record the initial values at each differenceing step. We need this in the 
            # meteadata to un-difference predicitons for inference. 
            if self.diff_count == 1:
                self.diff_1_data = self.data.to_dict()
            if self.diff_count == 2:
                self.diff_2_data = self.data.to_dict()
                
            self.log_to_file("-"*20)
        
        #Now that we have stationary data we have to pick the best p,d, and q values based on auto-correlation. This I did previously by looking at the plots, but a grid search works just as well.
        # self.data.reset_index(inplace=True)

        best_order, best_model = self.grid_search_arima()
        if self.verbose:
            self.log_to_file(str(best_model.summary())) # Uncomment this if you want to see the best model summary
            

        df = self.data.reset_index(drop = True).to_frame()
        df['forecast'] = pd.Series(best_model.predict()).reset_index(drop = True)
        if self.visualize: 
            df.plot(figsize=(12, 8))

        

        #Now we can look at performance values

        rmse = np.sqrt(np.mean((df['forecast']-df[self.feature_column]) ** 2))
        mask = df[self.feature_column] != 0
        mape = abs(((df['forecast'][mask] - df[self.feature_column][mask]) / df[self.feature_column][mask]).mean())
        mae_val = mae(df[self.feature_column], df['forecast'])

        # Create metadata json for model

        # Create metadata for model
        metadata = {
                "performance": {
                    "Model": 'ARIMA',
                    "AIC": best_model.aic,
                    "BIC": best_model.bic,
                    "RMSE": rmse,
                    "MAE": mae_val,
                    "MAPE": mape
                },
                "metadata": {
                    "diff_count": self.diff_count
                },
                "K-order data": {
                    "diff_0_data": self.convert_keys_to_str(self.orig_data.to_dict()),
                    "diff_1_data": self.diff_1_data,
                    "diff_2_data": self.diff_2_data,
                }
            }
        
        return metadata, self.data, best_model



        

            

