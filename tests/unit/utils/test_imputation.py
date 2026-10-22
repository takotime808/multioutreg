# Copyright (c) 2026 takotime808

import numpy as np
import pandas as pd

from multioutreg.utils.imputation import drop_unusable_rows, impute_train_test


def test_drop_unusable_rows_drops_missing_targets_never_imputes_them():
    df = pd.DataFrame({
        "x0": [1.0, np.nan, 3.0, 4.0],
        "x1": [1.0, 2.0, np.nan, 4.0],
        "y":  [1.0, 2.0, 3.0, np.nan],
    })
    out = drop_unusable_rows(df, output_cols=["y"], cols_to_drop_rows=["x1"])
    # row 3 (missing target) and row 2 (missing drop-col) go; row 1 (missing
    # an imputable input) stays, untouched, for imputation after the split.
    assert out.index.tolist() == [0, 1]
    assert np.isnan(out.loc[1, "x0"])
    assert df.shape == (4, 3)  # input not modified


def test_impute_train_test_is_not_influenced_by_test_rows():
    # Regression: imputation was fitted on the full frame before the split, so
    # test rows shaped the values filled into training rows.
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [np.nan, 1.1]])
    X_test_a = np.array([[1.0, 1.0]])
    X_test_b = np.array([[100.0, 1.05]])  # would be the nearest neighbour if used
    tr_a, _ = impute_train_test(X_train, X_test_a, n_neighbors=1)
    tr_b, _ = impute_train_test(X_train, X_test_b, n_neighbors=1)
    np.testing.assert_allclose(tr_a, tr_b)
    assert tr_a[3, 0] == 1.0  # nearest *training* row on x1 is [1, 1]


def test_impute_train_test_fills_test_from_training_neighbours():
    X_train = np.array([[0.0, 0.0], [10.0, 10.0]])
    X_test = np.array([[np.nan, 9.0]])
    _, te = impute_train_test(X_train, X_test, n_neighbors=1)
    assert te[0, 0] == 10.0


def test_impute_train_test_uses_other_columns_not_column_mean():
    # Previously only the selected column was passed to the imputer, so a lone
    # column with gaps could only be mean-filled.
    X_train = np.array([[0.0, 0.0], [0.1, 0.1], [9.9, 9.9], [10.0, 10.0], [np.nan, 9.95]])
    tr, _ = impute_train_test(X_train, X_train[:1], n_neighbors=2)
    assert abs(tr[4, 0] - 9.95) < 1e-9  # mean of the two nearest rows, not ~5


def test_impute_train_test_passthrough_and_shape():
    X = np.arange(6.0).reshape(3, 2)
    tr, te = impute_train_test(X, X)
    np.testing.assert_array_equal(tr, X)
    X_nan_col = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    tr, te = impute_train_test(X_nan_col, X_nan_col)
    assert tr.shape == X_nan_col.shape and te.shape == X_nan_col.shape
