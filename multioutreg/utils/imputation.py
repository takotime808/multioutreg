# Copyright (c) 2026 takotime808

"""Utilities for detecting and handling missing values in DataFrames."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer


def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of missing values per column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by column name with columns ``missing_count`` and
        ``missing_pct``.  Only columns that have at least one missing value
        are included.  Returns an empty DataFrame when no values are missing.
    """
    counts = df.isnull().sum()
    pcts = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"missing_count": counts, "missing_pct": pcts})
    return summary[summary["missing_count"] > 0]


def apply_imputation(
    df: pd.DataFrame,
    cols_to_impute: list[str],
    cols_to_drop_rows: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Clean a DataFrame by KNN-imputing selected columns and dropping rows elsewhere.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified in-place).
    cols_to_impute : list[str]
        Columns where missing values should be filled using KNN imputation.
    cols_to_drop_rows : list[str]
        Columns where any row with a missing value should be dropped entirely.
    n_neighbors : int, optional
        Number of nearest neighbors used by :class:`~sklearn.impute.KNNImputer`.
        Default is 5.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of ``df``.
    """
    df = df.copy()
    if cols_to_impute:
        imp = KNNImputer(n_neighbors=n_neighbors)
        df[cols_to_impute] = imp.fit_transform(df[cols_to_impute])
    if cols_to_drop_rows:
        df = df.dropna(subset=cols_to_drop_rows)
    return df
