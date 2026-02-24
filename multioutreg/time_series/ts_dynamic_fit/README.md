# TS-Dynamic Fit

A comprehensive, reusable data engineering and machine learning pipeline for automated time series forecasting with model selection, training, and evaluation.

## Project Overview

This pipeline automatically selects and trains optimal time series forecasting models for standardized datasets. It includes data preprocessing, feature engineering, model selection, training, evaluation, and deployment-ready output.

## Features

- **Data Processing**: Automated data validation, cleaning, and preprocessing
- **Feature Engineering**: Time-based features, lag variables, rolling statistics
- **Model Selection**: Automated comparison of multiple forecasting algorithms
- **Evaluation**: Comprehensive performance assessment with multiple metrics
- **Forecasting**: Generate predictions with confidence intervals
- **Visualization**: Automated chart generation for results

## Supported Models

- ARIMA
- SARIMA 
- LSTM
- 
- Additional models can be easily added

## Quick Start



### 1. Installation

`source .venv/bin/activate` or `.venv/Scripts/activate`

`uv sync`

If you need to run the jupyter notebook and dont see the .venv kernal, try the following:

`python -m ipykernel install --user --name=myvenvkernel --display-name "Python (.venv)"`

You will also need a .env file with the following information in order to pull from S3: 

```
AWS_ACCESS_KEY_ID=$key
AWS_SECRET_ACCESS_KEY=$secret
AWS_DEFAULT_REGION=$region
```

### 2. Command Line Usage

# Run on your data
`python main.py`

You can customize the run using the following arguments:
```
| Argument         | Default         | Type      | Description |
|------------------|-----------------|-----------|-------------|
| `--pull_new_data`| `False`         | flag      | Pull new data from S3 before analysis. Use this flag to refresh input data. |
| `--datetime_col` | `"index"`       | string    | Name of the datetime column in your dataset. |
| `--target_col`   | `"customers"`   | string    | Name of the target column to be predicted. |
| `--freq`         | `"1D"`          | string    | Time frequency for resampling (e.g., `"1D"` for daily, `"1H"` for hourly). |
| `--verbose`      | `False`         | flag      | Enable verbose logging for detailed output. |
| `--log_to_file`  | `"pipeline.log"`| string    | Filename where logs will be saved. |
```
### Examples

# Run with new data pulled from S3 and verbose logging:

`python main.py --pull_new_data --verbose`

## Data Format Requirements

Input data should be a CSV file with:
- A datetime column (any standard format)
- A numeric target variable column
- Additional feature columns (optional)

Example:
```csv
timestamp,sales,checks,labor_hours
2024-01-01 00:00:00,150.5,3,2.5
2024-01-01 00:15:00,200.0,4,2.5
...
```

## Output Files

- `trained_model.pkl`: Serialized best model and preprocessing pipeline
- `metadata.json`: Model specific metadata needed for loading and inference
- `pipeline.log`: Model comparison and evaluation metrics, only if verbose=True
- `scaler.pkl`: Scaler object needed for LSTM forcasting

## Model Performance Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **AIC/BIC**: Information criteria for ARIMA family models

## Architecture

```
root:.
│   ControlTower.ipynb
│   main.py
│   notes.txt
│   pipeline.log
│   __init__.py
│
├───algs
│   │   arima.py
│   │   lstm.py
│   │   sarima.py
│   │   __init__.py
│   
├───data_handling
│   │   Analyzer.py
│   │   DataProcessor.py
│   │   ingest.py
│   │   PullData.py
│   │   __init__.py
│   │
│   ├───data
│   │   │   OrderDetails_2025_06_20-2025_09_20.csv
│   │   │   Shifts_Closed_2024_06_06-2024_12_06.csv
│   │   │   sp_hourly_sales.xlsx
│   │   │   sp_hourly_sales_6month.csv
│   │   │   TimeEntries_2024_06_06-2024_12_06.csv
│   │
│   ├───old
│   │       OrderDetails_2024_09_06-2024_12_06.csv
│   │
│   ├───preds
│   │       weekly_2024_12_07-2024_12_20.csv
│
├───logs
│       pipeline.log
│
├───model registry
│   │   LSTM.pt
│   │   metadata.json
│   │
│   └───example
│       │   ARIMA.pkl
│       │   LSTM.pt
│       │   metadata.json
│       │
├───src
│   │   ForecastAutoIntervals.py
│   │   infer.py
│   │   Ranker.py
│   │   TSDataLoader.py
│   │   visualize.py
│   │   __init__.py
│   │
└───temp
        scaler.pkl
```

## Performance Requirements

- Supports datasets up to 50,000 time points
- Model training completion within 10 minutes on standard hardware
- Handles frequencies from minutes to daily

## Troubleshooting

### Common Issues

1. **Memory Error**: Reduce feature engineering complexity or use sampling
2. **Convergence Issues**: Adjust parameters, analyze pre-processing pipeline results, look at shift() values
3. **Poor Big(O) Performance**: Make sure p,q ranges are narrow, nn training loops are reasonable, and joblib is working
4. **SARIMA MAPE inf or NaN**: Parameter space is too large for the complexity of the data; reduce ranges for p,q,P,Q
   

## License

Author: Iapetus AI LLC
Date: September 2025