# Copyright (c) 2025 takotime808

import numpy as np
import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit import DataProcessor


def _make_ts_df(n=60, freq="1D"):
    dates = pd.date_range("2023-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "date": dates,
        "sales": np.abs(rng.standard_normal(n)) * 10 + 50,
    })


def test_data_processor_load_dataframe():
    """load_data() accepts a DataFrame directly."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    result, shape0 = dp.load_data(df)
    assert result is not None
    assert shape0 == len(df)


def test_data_processor_validate_returns_dict():
    """validate_data() returns a dict with shape and null_counts."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    v = dp.validate_data(datetime_col="date", target_col="sales")
    assert "shape" in v
    assert "null_counts" in v
    assert "target_stats" in v


def test_data_processor_no_file_writes(tmp_path, monkeypatch):
    """DataProcessor must not write any files during preprocessing."""
    monkeypatch.chdir(tmp_path)
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    dp.validate_data(datetime_col="date", target_col="sales")
    dp.preprocess_data(datetime_col="date", target_col="sales", freq="1D")

    created = list(tmp_path.iterdir())
    assert created == [], f"Unexpected files created: {created}"


def test_data_processor_engineer_features_adds_lag_columns():
    """engineer_features() produces lag_{n} and mean_{n} columns."""
    df = _make_ts_df(n=120, freq="1D")
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    dp.preprocess_data(datetime_col="date", target_col="sales", freq="1D")
    result = dp.engineer_features(target_col="sales")

    assert any("lag_" in c for c in result.columns)
    assert any("mean_" in c for c in result.columns)


def test_data_processor_engineer_features_no_nans():
    """After engineer_features(), there should be no NaN values."""
    df = _make_ts_df(n=120, freq="1D")
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    dp.preprocess_data(datetime_col="date", target_col="sales", freq="1D")
    result = dp.engineer_features(target_col="sales")

    assert result.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# CSV and file loading branches
# ---------------------------------------------------------------------------

def test_data_processor_load_csv(tmp_path):
    """load_data() accepts a path to a CSV file."""
    df = _make_ts_df()
    csv_path = tmp_path / "ts.csv"
    df.to_csv(csv_path, index=False)

    dp = DataProcessor(verbose=False)
    result, shape0 = dp.load_data(str(csv_path), format="csv")
    assert result is not None
    assert shape0 == len(df)


def test_data_processor_load_unsupported_format_returns_none(tmp_path):
    """load_data() with unsupported format returns None (exception caught internally)."""
    dp = DataProcessor(verbose=False)
    result = dp.load_data("fake_path.json", format="json")
    assert result is None


# ---------------------------------------------------------------------------
# validate_data edge cases
# ---------------------------------------------------------------------------

def test_data_processor_validate_no_data():
    """validate_data() returns error dict when no data has been loaded."""
    dp = DataProcessor(verbose=False)
    result = dp.validate_data()
    assert "error" in result


def test_data_processor_validate_no_datetime_no_target():
    """validate_data() without optional args still returns shape and null_counts."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    result = dp.validate_data()
    assert "shape" in result
    assert "null_counts" in result
    assert "datetime_range" not in result
    assert "target_stats" not in result


def test_data_processor_validate_datetime_range_present():
    """validate_data() with datetime_col includes datetime_range."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    result = dp.validate_data(datetime_col="date")
    assert "datetime_range" in result
    assert "start" in result["datetime_range"]
    assert "end" in result["datetime_range"]


def test_data_processor_validate_duplicates_key():
    """validate_data() includes duplicates count."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=False)
    dp.load_data(df)
    result = dp.validate_data()
    assert "duplicates" in result


# ---------------------------------------------------------------------------
# preprocess_data and engineer_features error paths
# ---------------------------------------------------------------------------

def test_data_processor_preprocess_no_data_raises():
    """preprocess_data() raises ValueError when no data has been loaded."""
    dp = DataProcessor(verbose=False)
    with pytest.raises(ValueError, match="No data loaded"):
        dp.preprocess_data(datetime_col="date", target_col="sales")


def test_data_processor_engineer_features_no_processed_data_raises():
    """engineer_features() raises ValueError when preprocess_data was not called."""
    dp = DataProcessor(verbose=False)
    with pytest.raises(ValueError, match="No processed data available"):
        dp.engineer_features(target_col="sales")


# ---------------------------------------------------------------------------
# verbose=True paths
# ---------------------------------------------------------------------------

def test_data_processor_verbose_load_dataframe():
    """verbose=True does not raise when loading a DataFrame."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=True)
    result, shape0 = dp.load_data(df)
    assert result is not None


def test_data_processor_verbose_validate():
    """verbose=True does not raise during validate_data."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=True)
    dp.load_data(df)
    result = dp.validate_data(datetime_col="date", target_col="sales")
    assert "shape" in result


def test_data_processor_verbose_preprocess():
    """verbose=True does not raise during preprocess_data."""
    df = _make_ts_df()
    dp = DataProcessor(verbose=True)
    dp.load_data(df)
    result = dp.preprocess_data(datetime_col="date", target_col="sales", freq="1D")
    assert result is not None


def test_data_processor_verbose_engineer_features():
    """verbose=True does not raise during engineer_features."""
    df = _make_ts_df(n=120, freq="1D")
    dp = DataProcessor(verbose=True)
    dp.load_data(df)
    dp.preprocess_data(datetime_col="date", target_col="sales", freq="1D")
    result = dp.engineer_features(target_col="sales")
    assert result is not None
