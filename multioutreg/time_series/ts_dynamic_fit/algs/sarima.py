# Copyright (c) 2025 takotime808
from sklearn.metrics import mean_absolute_error as mae
from statsmodels.tsa.stattools import adfuller, acf, pacf
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from joblib import Parallel, delayed
import warnings
import logging
from typing import Optional, Tuple, Any, Dict, Sequence, Union

logging.basicConfig(level=logging.INFO, format='%(message)s')

warnings.filterwarnings("ignore")


class SARIMA:
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

    def log_to_file(self, log: str) -> None:
        logging.info(log)
        with open(f"logs/{self.log_filename}", 'a') as f:
            f.write(log + '\n')

        
    def make_stationary(self, df):
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
        self.diff_count += 1

        if adf[1] > 0.05:
            if self.verbose: 
                self.log_to_file(f"The p-value: {adf[1]} implies the series is non-stationary and requires differencing.")
            return adf
        else:
            if self.verbose:
                self.log_to_file("Data is stationary.")
            self.stationary = True
            return adf
    
    def fit_sarima(self, p, q, P, Q, s_value):
        try:
            model = sm.tsa.statespace.SARIMAX(
                self.data,
                order=(p, 0, q),
                seasonal_order=(P, 0, Q, s_value)
            )
            results = model.fit(disp=False)
            return (p, 0, q, P, 0, Q, s_value), results.aic, results
        except:
            return (p, 0, q, P, 0, Q, s_value), np.inf, None
            
    def grid_search_sarima(self, p_values=range(0,2), q_values=range(0,2), P_values=range(0,2), Q_values=range(0,2), s_value=0):
        """
            Perform a grid search to find the optimal SARIMA model order (p, d, q, s) for a given time series.
            For our use "d" will always be 0 because we are only using this on data we previously made stationary. 
            The s value (seaosnal period) has to have its own detection run, which is why it is passed here as an argument. 
        
            This function iterates through all combinations of provided AR (p) and MA (q) orders,
            fits an SARIMA model for each (p, 0, q, s) quartet, and selects the order with the lowest AIC score.
            It returns the optimal model order along with the fitted model instance.
        
            Args:
                X (array-like or pandas.Series): The univariate time series data to fit.
                p_values (iterable of int): Sequence of candidate AR (autoregressive) orders to try.
                q_values (iterable of int): Sequence of candidate MA (moving average) orders to try.
        
            Returns:
                tuple:
                    best_order (tuple of int): The (p, q) order with the lowest AIC value found.
                    best_model (statsmodels SARIMAResults): The fitted ARIMA model instance corresponding to best_order.
            Raises:
                Any exceptions raised by statsmodels SARIMA on invalid configurations will be caught and skipped.
        """
                
        best_aic = np.inf
        best_order = None
        best_model = None

        results = Parallel(n_jobs=-1)(
            delayed(self.fit_sarima)(p, q, P, Q, s_value)
            for p in p_values
            for q in q_values
            for P in P_values
            for Q in Q_values
            )

        for (order, aic, model) in results:
            if model is not None and aic < best_aic:
                best_aic = aic
                best_order = order
                best_model = model
        
        return best_order, best_model

    def detect_seasonal_period(self, data, max_lag=None, threshold=0.3):
        """
        Automatically find likely seasonality period based on ACF peak.
        
        Args:
            data (array-like): Time series data 
            max_lag (int, optional): Maximum lag to check. Default: len(data)//2.
            threshold (float): Minimum ACF value considered as a significant peak.
        
        Returns:
            s_value (int): Most likely seasonality period (lag of first significant ACF peak).
        """
        if max_lag is None:
            max_lag = min(len(data)//2, 100)
        acf_vals = acf(data, nlags=max_lag, fft=True)
        # Ignore lag=0
        for lag in range(1, len(acf_vals)):
            if acf_vals[lag] > threshold:
                return lag
        return 0  # No strong seasonality detected
                
    def run(self) -> Tuple[Dict[str, Any], pd.DataFrame, Any]:
        """
        Runs the SARIMA parameter search, performs differencing for stationarity,
        finds the best SARIMA model order, evaluates model performance, and returns results.
        """
        self.log_to_file("#"*40)
        self.log_to_file("Running SARIMA Parameter Search...")
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
        
        s = self.detect_seasonal_period(self.data)
        best_order, best_model = self.grid_search_sarima(s_value=s)
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


        # Create metadata for model
        metadata = {
                "performance": {
                    "Model": 'SARIMA',
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
                    "diff_0_data": self.orig_data.to_dict(),
                    "diff_1_data": self.diff_1_data,
                    "diff_2_data": self.diff_2_data,
                }
            }

        return metadata, self.data, best_model



        

            

