# Copyright (c) 2025 takotime808

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

from multioutreg.figures.uncertainty_toolbox_extension import (
    plot_uct_intervals_ordered_multioutput,
)

@pytest.fixture
def get_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    x = np.array([4, 5, 6.5])
    return y_pred, y_std, y_true, x


def test_plot_uct_intervals_ordered_multioutput_returns(get_test_set):
    """Test multioutput wrapper returns correct axes."""
    y_pred, y_std, y_true, _ = get_test_set
    # create simple 2-output data by repeating arrays
    y_pred = np.stack([y_pred, y_pred], axis=1)
    y_std = np.stack([y_std, y_std], axis=1)
    y_true = np.stack([y_true, y_true], axis=1)
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert isinstance(axes, list)
    assert len(axes) == 2
    for ax in axes:
        assert isinstance(ax, matplotlib.axes.Axes)


@pytest.fixture
def example_data():
    np.random.seed(42)
    y_pred = np.random.randn(20, 3)
    y_std = np.abs(np.random.randn(20, 3)) + 0.1
    y_true = np.random.randn(20, 3)
    return y_pred, y_std, y_true


def test_shape_check_raises(example_data):
    y_pred, y_std, y_true = example_data
    # Change shape to cause error
    y_bad = y_pred[:, :2]
    with pytest.raises(ValueError, match="same shape"):
        plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_bad)


def test_dimension_check_raises(example_data):
    y_pred, y_std, y_true = example_data
    with pytest.raises(ValueError, match="must be 2D arrays"):
        # Make y_pred 3D
        plot_uct_intervals_ordered_multioutput(y_pred[..., None], y_std, y_true)


def test_1d_input_works():
    np.random.seed(0)
    y_pred = np.random.randn(30)
    y_std = np.abs(np.random.randn(30)) + 0.1
    y_true = np.random.randn(30)
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert isinstance(axes, list)
    assert hasattr(axes[0], "plot")  # matplotlib Axes


def test_2d_input_axes_count(example_data):
    y_pred, y_std, y_true = example_data
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true)
    assert isinstance(axes, list)
    assert len(axes) == y_pred.shape[1]
    for ax in axes:
        assert hasattr(ax, "plot")


def test_ax_list_usage(example_data):
    y_pred, y_std, y_true = example_data
    fig, axs = matplotlib.pyplot.subplots(1, y_pred.shape[1])
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, ax_list=axs)
    assert all(ax in axs for ax in axes)


def test_savefig_creates_file(tmp_path, example_data):
    y_pred, y_std, y_true = example_data
    save_path = tmp_path / "testfig.png"
    plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, savefig=str(save_path))
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_custom_suptitle(example_data):
    y_pred, y_std, y_true = example_data
    custom_title = "My Custom Title"
    # Should not raise error
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, suptitle=custom_title)
    assert isinstance(axes, list)


def test_ylims_applied(example_data):
    y_pred, y_std, y_true = example_data
    ylims = (-1, 1)
    axes = plot_uct_intervals_ordered_multioutput(y_pred, y_std, y_true, ylims=ylims)
    for ax in axes:
        ylow, yhigh = ax.get_ylim()
        assert ylow <= ylims[0] + 1e-6
        assert yhigh >= ylims[1] - 1e-6
