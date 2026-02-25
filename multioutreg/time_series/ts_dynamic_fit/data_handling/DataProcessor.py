# Copyright (c) 2025 takotime808
import pandas as pd
import numpy as np
import warnings
import logging
import pprint
from typing import Optional, Dict, Any, Union

warnings.filterwarnings('ignore')
_logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data engineering pipeline for time series preprocessing.
    Handles data ingestion, validation, cleaning, and feature engineering.
    """
    
    def __init__(self, verbose: bool = False, log_filename: str = "pipeline.log") -> None:
        self.data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.verbose: bool = verbose
        self.log_filename: str = log_filename
        
    def log_to_file(self, log: str) -> None:
        _logger.info(log)
        
    def load_data(
        self, 
        data: Union[str, pd.DataFrame], 
        format: str = 'csv'
    ) -> Optional[pd.DataFrame]:
        """
        Load time series data from various formats or directly from a DataFrame.
    
        Args:
            data (str or pd.DataFrame): Path to the data file or a DataFrame
            format (str): Format of the file ('csv', 'excel'). Ignored if DataFrame.
    
        Returns:
            pd.DataFrame: Loaded data
            future_pred: Loaded data shape at dimension 0 (needed for prediction intervals in deep models)
        """
        import pandas as pd
        try:
            if isinstance(data, pd.DataFrame):
                self.data = data
                if self.verbose:
                    self.log_to_file(f"DataFrame accepted directly. Shape:{self.data.shape}\n")
            else:
                if format.lower() == 'csv':
                    self.data = pd.read_csv(data)
                elif format.lower() in ['excel', 'xlsx']:
                    self.data = pd.read_excel(data)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                if self.verbose:
                    self.log_to_file(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data, self.data.shape[0]
    
        except Exception as e:
            self.log_to_file(f"Error loading data: {str(e)}")
            return None
    
    def validate_data(
        self, 
        datetime_col: Optional[str] = None, 
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform automated data quality checks.
        
        Args:
            datetime_col (str): Name of datetime column
            target_col (str): Name of target variable column
        
        Returns:
            dict: Validation results
        """
        validation_results: Dict[str, Any] = {}
        
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Basic validation
        validation_results['shape'] = self.data.shape
        validation_results['null_counts'] = self.data.isnull().sum().to_dict()
        validation_results['duplicates'] = self.data.duplicated().sum()
        
        # Datetime validation
        if datetime_col:
            try:
                self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
                validation_results['datetime_range'] = {
                    'start': str(self.data[datetime_col].min()),
                    'end': str(self.data[datetime_col].max())
                }
            except Exception as e:
                validation_results['datetime_error'] = str(e)
        
        # Target variable validation
        if target_col:
            validation_results['target_stats'] = {
                'mean': self.data[target_col].mean(),
                'std': self.data[target_col].std(),
                'min': self.data[target_col].min(),
                'max': self.data[target_col].max(),
                'zeros': (self.data[target_col] == 0).sum()
            }
        if self.verbose:
            _logger.info("Validation results:\n%s", pprint.pformat(validation_results))
        
        return validation_results
    
    def preprocess_data(
        self, 
        datetime_col: str, 
        target_col: str, 
        freq: str = '15min'
    ) -> pd.DataFrame:
        """
        Preprocess time series data with transformations and cleaning.
        
        Args:
            datetime_col (str): Name of datetime column
            target_col (str): Name of target variable column
            freq (str): Resampling frequency
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Work with a copy
        df = self.data.copy()
        
        # Convert datetime column
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Set datetime as index
        df.set_index(datetime_col, inplace=True)
        
        # Handle missing values
        df[target_col] = df[target_col].fillna(0)  # Fill with 0 for sales data
        
        # Remove outliers using IQR method
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them to preserve time structure
        df[target_col] = df[target_col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        # Ensure regular frequency
        df = df.asfreq(freq, fill_value=0)
        
        self.processed_data = df
        if self.verbose: 
            self.log_to_file(f"Data preprocessed. Final shape: {df.shape}")
        return df
    
    def engineer_features(
        self, 
        target_col: str
    ) -> pd.DataFrame:
        """
        Create time series features for model training. 
        I think this will be helpful for future analysis as all the info is in one place, and the same data can be used. 
        
        Args:
            target_col (str): Name of target variable column
        
        Returns:
            pd.DataFrame: Data with engineered features
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        df = self.processed_data.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int) # This might be scuffed for SomebodyPeople because I think weekends are already removed... but here for future data formats 
        
        # Lag features
        for lag in [1, 2, 3, 4, 24, 96]:  # 15min, 30min, 45min, 1hr, 6hr, 24hr lags
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling window features
        for window in [4, 8, 24, 96]:  # 1hr, 2hr, 6hr, 24hr windows
            df[f'{target_col}_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()
        
        # Fill NaN values created by lag and rolling features
        df = df.bfill().ffill().fillna(0)
        
        self.processed_data = df
        if self.verbose:
            self.log_to_file(f"Features engineered. Shape: {df.shape}")
        return df

