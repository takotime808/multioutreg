# Copyright (c) 2025 takotime808
# Copyright (c) 2025 takotime808

from src.TSDataLoader import *
from algs.arima import *
from algs.sarima import *
from algs.lstm import *
from src.Ranker import *
from data_handling.DataProcessor import *
from data_handling.PullData import *

from src.visualize import visualize_model
import json
from src.infer import infer, plot_inference
import argparse
from pathlib import Path

def main(args):

    if args.pull_new_data:
        pull_data()

    # To prevent confusion when reading the logs, this will erase the previous runs logs before writing new logs
    with open("pipeline.log", "w") as f:
        pass 

    # Also remove old model reg objects and configs
    [f.unlink() for f in Path("model registry").iterdir() if f.is_file()]

        
    verbose = args.verbose
    log_filename = args.log_to_file
    
    TSD = TSDataLoader("data_handling/data")
    tsdata = TSD.get() # Get the time series data
    
    # 1. Initialize processor
    processor = DataProcessor(verbose=verbose)
    
    # 2. Load data
    data, future_pred = processor.load_data(tsdata)
    
    # 3. Validate data
    validation_results = processor.validate_data(
        datetime_col=args.datetime_col,    # Replace with whatever the actual datetime column is
        target_col=args.target_col   # Replace with whatever the target variable is
    )
    
    # 4. Preprocess data
    processed_data = processor.preprocess_data(
        datetime_col=args.datetime_col,      # Datetime column
        target_col=args.target_col,    # Target variable
        freq='1D'                  # Time frequency (15min, 1H, 1D, etc.)
    )
    
    # 5. Engineer features
    featured_data = processor.engineer_features(
        target_col=args.target_col        # Ttarget variable
    )
    
    # Then define the types of analysis you want to perform
    data = processed_data
    feature_column = args.target_col
    
    arima= ARIMA(data, feature_column, verbose=verbose, log_filename=log_filename)
    sarima = SARIMA(data, feature_column, verbose=verbose)
    lstm = LSTM(data, feature_column=feature_column, fut_pred=future_pred, train_window=10, verbose=verbose)

    
    # Add them to the candidate model list
    p1, d1, m1, = arima.run() # performance dictionary 1, data object 1, model object 1
    p2, d2, m2, = sarima.run() # performance dictionary 2, data object 2, model object 2
    p3, d3, m3, = lstm.run() # performance dictionary 3, data object 3, model object 3
    
    # Store your performance and model objects
    perf_dicts = [p1,p2,p3]
    model_objs = [m1,m2,m3]
    
    candidates = pd.DataFrame(perf_dicts)
    
    # And see which model has the highest average performance on our data
    best_model = Ranker(df=candidates, verbose=verbose).get_best()
    print(best_model)
    
    # Visuals, if verbose
    if verbose: 
        if best_model == 'ARIMA':
            visualize_model(m1, d1, target_col=args.target_col,)
        if best_model == 'SARIMA':
            visualize_model(m2, d2, target_col=args.target_col,)
        elif best_model == 'LSTM':
            pass
    
    # Serialize best model
    idx = [p['performance']['Model'] for p in perf_dicts].index(best_model)
    best_model_obj = model_objs[idx]
    metadata = perf_dicts[idx]['metadata']
    
    # Save Model Metedata for future reference
    if perf_dicts[idx]['performance']["Model"] == "ARIMA" or perf_dicts[idx]['performance']["Model"] == "SARIMA":
        with open(f'Model Registry/{str(best_model)}.pkl', 'wb') as f:
            pickle.dump(best_model_obj, f)
    
        with open('Model Registry/metadata.json', 'w') as f:
            json.dump(perf_dicts[idx], f)
    
    if perf_dicts[idx]['performance']["Model"] == "LSTM":
        torch.save(best_model_obj.state_dict(), "Model Registry/LSTM.pt")
    
        with open('Model Registry/metadata.json', 'w') as f:
            json.dump(perf_dicts[idx], f)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Time Series Model Pipeline")
    parser.add_argument("--pull_new_data", default=False, action="store_true", help="Should new data be pulled from S3 for analysis")
    parser.add_argument("--datetime_col", default="index", help="Name of the datetime column")
    parser.add_argument("--target_col", default="customers", help="Name of the target column")
    parser.add_argument("--freq", default="1D", help="Time frequency (e.g., 1D, 1H)")
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose output and logging")
    parser.add_argument("--log_to_file", default="pipeline.log", type=str, help="Log filename")

    args = parser.parse_args()
    main(args)


