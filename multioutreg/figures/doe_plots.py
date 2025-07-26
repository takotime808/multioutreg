# Copyright (c) 2025 takotime808

"""DoE Pair Plots: Visualization utility for generating PairGrid plots of numerical variables from a DataFrame."""

import pandas as pd
import seaborn as sns
# from seaborn.axisgrid import PairGrid
from typing import List


def make_doe_plot(df: pd.DataFrame, numeric_cols: List[str]) -> sns.PairGrid:
    """
    Generate a Seaborn PairGrid plot for the selected numeric columns of a DataFrame.
    
    The plot includes:
        - Scatter plots in the upper triangle
        - Kernel density estimates in the lower triangle
        - Histograms with KDE on the diagonal

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        numeric_cols (List[str]): List of column names to include in the pair plot.

    Returns:
        PairGrid: A Seaborn PairGrid object for further customization or rendering.
    """
    sampled_df = df[numeric_cols].sample(min(200, len(df)), random_state=0)

    # Create the PairGrid
    g = sns.PairGrid(sampled_df)

    # Upper triangle: scatterplot
    g.map_upper(sns.scatterplot, s=15)

    # Lower triangle: kdeplot
    g.map_lower(sns.kdeplot, fill=True, cmap="Blues")

    # Diagonal: hist
    g.map_diag(sns.histplot, kde=True)
    return g