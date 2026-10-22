# Copyright (c) 2025 takotime808

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

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


def gaussian_uncertainty_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    nominal: float = 0.95,
) -> Dict[str, float]:
    """Average (over outputs) calibration metrics of the Gaussian ``mean ± std``.

    Returns sharpness (mean predicted std), empirical coverage of the central
    ``nominal`` Gaussian interval, miscoverage ``|nominal - coverage|`` and,
    when every std is positive, the mean Gaussian negative log-likelihood.
    """
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_std = np.asarray(y_std, dtype=float)
    err = np.abs(y_test - y_pred)
    z = norm.ppf(0.5 + nominal / 2)
    coverage = float(np.mean(err <= z * y_std))
    pct = f"{nominal:.0%}"
    out = {
        "Sharpness (mean predicted std)": float(np.mean(y_std)),
        f"Empirical coverage of {pct} Gaussian interval": coverage,
        f"Miscoverage at {pct} (|nominal − empirical|)": abs(nominal - coverage),
    }
    # NLL is undefined for zero std (models without native uncertainty).
    if np.all(y_std > 0):
        out["Gaussian NLL (mean per point)"] = float(
            np.mean(-norm.logpdf(y_test, loc=y_pred, scale=y_std))
        )
    return out
