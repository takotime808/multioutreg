# Copyright (c) 2025 takotime808

import pandas as pd

def flatten_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens a DataFrame with columns containing dictionaries (as returned by uncertainty_toolbox)
    into a DataFrame with one column per metric.

    For each column (except 'output') where each cell is a dict, this function extracts all key-value
    pairs, creating a new column for each unique key. If a key is repeated in multiple columns, the new
    column is prefixed with the original column name to avoid naming conflicts.

    Args:
        df (pd.DataFrame): Input DataFrame with columns of dicts (and an 'output' column).

    Returns:
        pd.DataFrame: A flattened DataFrame where each metric is a separate column, with 'output' preserved.

    Example:
        Input columns: ['accuracy', 'sharpness', 'output'], where 'accuracy' and 'sharpness' are dicts
        Output columns: ['mae', 'rmse', ..., 'sharp', ..., 'output']
    """
    # Start with 'output' as the index/column
    output_col = df['output']
    flat_dicts = {}

    for col in df.columns:
        if col == 'output':
            continue
        # Each item in this column is a dict
        for key in df[col][0].keys():
            # Make a new column name
            new_col = key
            # Avoid name conflicts by prefixing
            if key in flat_dicts:
                new_col = f"{col}_{key}"
            flat_dicts[new_col] = [row.get(key, None) for row in df[col]]

    # Make a new flat DataFrame
    flat_df = pd.DataFrame(flat_dicts)
    flat_df['output'] = output_col
    return flat_df