# Copyright (c) 2025 takotime808
import pandas as pd
import numpy as np
import logging
from typing import Any
import pickle

logging.basicConfig(level=logging.INFO)


class Ranker:
    """Ranks models in a DataFrame using error metrics and provides the best model.

    Attributes:
        metrics (list[str]): Error metrics used for ranking.
        all_models (pd.DataFrame): DataFrame containing model results.
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = False) -> None:
        """
        Initializes the Ranker with models and verbosity preference.

        Args:
            df (pd.DataFrame): DataFrame containing model statistics and metrics.
            verbose (bool): If True, prints detailed ranking info.
        """
        self.metrics = ['RMSE', 'MAE', 'MAPE']  # Error metrics for ranking
        self.all_models = df
        self.verbose = verbose


    def get_best(self) -> str:
        """
        Ranks all models using RMSE, MAE, MAPE and returns the best model name.

        Returns:
            Any: The best model's identifier/name from the DataFrame.
        """
        # Build dataframe
        df = self.all_models['performance'].apply(pd.Series)
        # df = pd.DataFrame(self.all_models['performance'])

        # Rank each metric (lower is better), 1 is best
        for m in self.metrics:
            df[f'{m}_Rank'] = df[m].rank(method='min')

        # Compute average rank across all metrics
        df['Avg_Rank'] = df[[f'{m}_Rank' for m in self.metrics]].mean(axis=1)

        # Sort and select best model(s)
        df = df.sort_values('Avg_Rank')
        best_model = df.iloc[0]['Model']
        cols = ['Model'] + self.metrics + [f'{m}_Rank' for m in self.metrics] + ['Avg_Rank']
        ranking_table = df[cols].copy()

        if self.verbose:
            logging.info("\n%s", df[['Model'] + self.metrics + [f'{m}_Rank' for m in self.metrics] + ['Avg_Rank']].to_string())
            logging.info("\nBest model: %s", best_model)
        return best_model

