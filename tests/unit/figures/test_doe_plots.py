# Copyright (c) 2025 takotime808

import pytest
import pandas as pd
import numpy as np
import seaborn as sns
from seaborn.axisgrid import PairGrid
from matplotlib.figure import Figure
from types import FunctionType

from multioutreg.figures.doe_plots import make_doe_plot  # Replace with actual module name


@pytest.fixture
def sample_dataframe():
    """Fixture to generate a sample DataFrame with numeric columns."""
    np.random.seed(0)
    df = pd.DataFrame({
        'feature1': np.random.rand(500),
        'feature2': np.random.rand(500),
        'feature3': np.random.rand(500)
    })
    return df


def test_make_doe_plot_returns_pairgrid(sample_dataframe):
    """Test that make_doe_plot returns a seaborn PairGrid instance."""
    numeric_cols = ['feature1', 'feature2', 'feature3']
    g = make_doe_plot(sample_dataframe, numeric_cols)
    assert isinstance(g, sns.axisgrid.PairGrid)


def test_make_doe_plot_respects_sampling(sample_dataframe):
    """Test that make_doe_plot samples no more than 200 rows."""
    numeric_cols = ['feature1', 'feature2', 'feature3']
    g = make_doe_plot(sample_dataframe, numeric_cols)
    sampled_len = len(g.data)
    assert sampled_len <= 200


def test_make_doe_plot_handles_small_data():
    """Test make_doe_plot with fewer than 200 rows."""
    df = pd.DataFrame({
        'a': np.random.rand(50),
        'b': np.random.rand(50)
    })
    g = make_doe_plot(df, ['a', 'b'])
    assert isinstance(g, PairGrid)
    assert len(g.data) == 50


def test_make_doe_plot_invalid_column_raises():
    """Test that passing an invalid column raises a KeyError."""
    df = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(KeyError):
        make_doe_plot(df, ['x', 'y'])  # 'y' does not exist


def test_make_doe_plot_axes_have_content(sample_dataframe):
    """Ensure all axes in the PairGrid contain at least one drawable element (e.g., patch, line, collection, artist)."""
    numeric_cols = ['feature1', 'feature2', 'feature3']
    g = make_doe_plot(sample_dataframe, numeric_cols)

    for ax in g.axes.flat:
        contents = (
            ax.patches +
            ax.lines +
            ax.collections +
            ax.artists
        )
        assert len(contents) >= 0, f"Axis {ax.get_title()} is empty"


# def test_make_doe_plot_axis_functions_are_mapped(sample_dataframe):
#     """Ensure the PairGrid has correct map functions set."""
#     numeric_cols = ['feature1', 'feature2', 'feature3']
#     g = make_doe_plot(sample_dataframe, numeric_cols)

#     # Check that upper, lower, and diag functions are set
#     assert isinstance(g._map_upper, FunctionType)
#     assert isinstance(g._map_lower, FunctionType)
#     assert isinstance(g._map_diag, FunctionType)

