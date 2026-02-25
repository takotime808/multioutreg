# Copyright (c) 2025 takotime808

import pandas as pd
import pytest

from multioutreg.time_series.ts_dynamic_fit.data_handling.Analyzer import (
    DataHolder,
    Analyzer,
)


def _make_orders_df(n=10):
    """Build a minimal orders DataFrame with a DatetimeIndex."""
    idx = pd.date_range("2023-01-02", periods=n, freq="h")  # Mon onwards
    servers = ["Alice", "Bob", "Carol"] * 4
    servers = servers[:n]
    guests = [2, 3, 4, 1, 5, 2, 3, 4, 1, 2][:n]
    return pd.DataFrame({"Server": servers, "# of Guests": guests}, index=idx)


# ---------------------------------------------------------------------------
# DataHolder
# ---------------------------------------------------------------------------

class TestDataHolder:

    def test_stores_orders(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        assert dh.orders is orders

    def test_optional_fields_default_none(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        assert dh.shifts is None
        assert dh.sales is None
        assert dh.timeentries is None

    def test_accepts_optional_fields(self):
        orders = _make_orders_df()
        shifts = pd.DataFrame({"x": [1, 2]})
        sales = pd.DataFrame({"y": [3, 4]})
        dh = DataHolder(orders=orders, shifts=shifts, sales=sales)
        assert dh.shifts is shifts
        assert dh.sales is sales


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class TestAnalyzer:

    def test_servers_per_day_returns_dataframe(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        result = analyzer.servers_per_day()
        assert isinstance(result, pd.DataFrame)
        assert "servers" in result.columns

    def test_customers_per_day_returns_dataframe(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        result = analyzer.customers_per_day()
        assert isinstance(result, pd.DataFrame)
        assert "customers" in result.columns

    def test_customers_per_day_sum_correct(self):
        """Total customers summed across all rows should equal sum in result."""
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        result = analyzer.customers_per_day()
        assert result["customers"].sum() == orders["# of Guests"].sum()

    def test_combine_relevant_data_has_both_columns(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        result = analyzer.combine_relevant_data()
        assert "servers" in result.columns
        assert "customers" in result.columns

    def test_combine_relevant_data_same_index(self):
        """servers_per_day and customers_per_day share the same index."""
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        result = analyzer.combine_relevant_data()
        # All rows should be present (inner join)
        assert len(result) > 0

    def test_restaurant_attribute_stored(self):
        orders = _make_orders_df()
        dh = DataHolder(orders=orders)
        analyzer = Analyzer(dh)
        assert analyzer.restaurant is dh
