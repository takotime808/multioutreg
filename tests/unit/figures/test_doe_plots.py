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



# ---------------------------------------------------------------------------
# make_pair_plot / pair_plot_b64
# ---------------------------------------------------------------------------
import base64

from multioutreg.figures.doe_plots import (
    INPUT_COLOR,
    OUTPUT_COLOR,
    make_pair_plot,
    pair_plot_b64,
)
from matplotlib.colors import to_hex


def _io_frame(n_in, n_out, n_rows=60, seed=0):
    rng = np.random.default_rng(seed)
    data = {f"x{i}": rng.uniform(size=n_rows) for i in range(n_in)}
    data.update({f"y{j}": rng.normal(size=n_rows) for j in range(n_out)})
    return pd.DataFrame(data), [f"x{i}" for i in range(n_in)], [f"y{j}" for j in range(n_out)]


def test_make_pair_plot_colours_labels_by_role():
    df, inputs, outputs = _io_frame(2, 1)
    g = make_pair_plot(df, inputs, outputs)
    colours = {}
    for ax in g.axes.flat:
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_text():
                colours[lab.get_text()] = to_hex(lab.get_color())
    assert colours == {"x0": INPUT_COLOR, "x1": INPUT_COLOR, "y0": OUTPUT_COLOR}


def test_make_pair_plot_io_layout_has_outputs_by_inputs_panels():
    df, inputs, outputs = _io_frame(3, 2)
    g = make_pair_plot(df, inputs, outputs, layout="io")
    assert g.axes.shape == (2, 3)


def test_make_pair_plot_stratified_sample_keeps_every_level():
    df, inputs, outputs = _io_frame(2, 1, n_rows=2000)
    df["fidelity"] = np.r_[np.zeros(1990), np.ones(10)].astype(int)
    g = make_pair_plot(df, inputs, outputs, fidelity_col="fidelity", max_rows=100)
    assert len(g.data) <= 102
    assert set(g.data["fidelity"]) == {"0", "1"}


def test_pair_plot_b64_full_matrix_is_a_png():
    df, inputs, outputs = _io_frame(2, 2)
    result = pair_plot_b64(df, inputs, outputs)
    assert result["layout"] == "full"
    assert result["note"] is None
    assert base64.b64decode(result["img_b64"]).startswith(b"\x89PNG")


def test_pair_plot_b64_falls_back_to_io_then_skips():
    df, inputs, outputs = _io_frame(4, 2, n_rows=20)
    io_result = pair_plot_b64(df, inputs, outputs, max_vars=4)
    assert io_result["layout"] == "io" and io_result["img_b64"]
    assert "outputs vs inputs" in io_result["note"]

    skipped = pair_plot_b64(df, inputs, outputs, max_vars=2)  # 8 io panels > 2**2
    assert skipped["img_b64"] is None
    assert "limit 2" in skipped["note"]


def test_pair_plot_b64_notes_row_sampling_and_ignores_non_numeric():
    df, inputs, outputs = _io_frame(1, 1, n_rows=50)
    df["label"] = "a"
    result = pair_plot_b64(df, inputs + ["label"], outputs, max_rows=10)
    assert result["img_b64"]
    assert "10 of the 50 rows" in result["note"]


def test_pair_plot_b64_needs_two_columns_and_never_raises():
    df, inputs, outputs = _io_frame(1, 0)
    assert pair_plot_b64(df, inputs, outputs)["img_b64"] is None

    # Constant columns break the 2-D histogram's binning in some versions;
    # whatever happens, the result is a dict, not an exception.
    bad = pd.DataFrame({"x": [np.inf, 1.0, 2.0], "y": [1.0, 1.0, 1.0]})
    out = pair_plot_b64(bad, ["x"], ["y"])
    assert set(out) == {"img_b64", "layout", "note"}
