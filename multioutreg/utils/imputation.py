# Copyright (c) 2026 takotime808

"""Utilities for detecting and handling missing values in DataFrames."""

from __future__ import annotations

import numpy as np
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

    .. note::
        The imputer is fitted on the whole frame, so do not use this before a
        train/test split: test rows would influence training values. For
        modeling, use :func:`drop_unusable_rows` before splitting and
        :func:`impute_train_test` after.

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


def drop_unusable_rows(
    df: pd.DataFrame,
    output_cols: list[str],
    cols_to_drop_rows: list[str],
) -> pd.DataFrame:
    """Drop rows that cannot be used for modeling, before any train/test split.

    Rows with a missing target are always dropped: an imputed target would be
    scored as ground truth. Rows with a missing value in ``cols_to_drop_rows``
    are dropped as well. Both decisions depend only on the row itself, so this
    is safe to do before splitting.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (not modified in-place).
    output_cols : list[str]
        Target columns; rows missing any of these are dropped.
    cols_to_drop_rows : list[str]
        Additional columns whose missing values cause the row to be dropped.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` without the unusable rows.
    """
    subset = list(dict.fromkeys(list(output_cols) + list(cols_to_drop_rows)))
    return df.dropna(subset=subset) if subset else df.copy()


def impute_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """KNN-impute missing input values without leaking test data.

    The imputer is fitted on the training inputs only and then applied to both
    splits, so test rows never influence the values filled into training rows
    (or each other). All input columns are used to find neighbours.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Input feature matrices, possibly containing NaN.
    n_neighbors : int, optional
        Number of nearest neighbours used by :class:`~sklearn.impute.KNNImputer`.

    Returns
    -------
    X_train_imputed, X_test_imputed : np.ndarray
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if not (np.isnan(X_train).any() or np.isnan(X_test).any()):
        return X_train, X_test
    imp = KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    return imp.fit_transform(X_train), imp.transform(X_test)
