# Copyright (c) 2025 takotime808

import pytest
import os
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multioutreg.figures.confidence_intervals import (
    plot_intervals_ordered,
    plot_intervals_ordered_multi,
    plot_confidence_interval,
    plot_multioutput_confidence_intervals,
    plot_regressormixin_confidence_intervals,
)


@pytest.fixture
def single_output_data():
    n = 20
    x = np.arange(n)
    y_true = np.sin(x)
    preds = np.sin(x) + 0.1 * np.random.randn(n)
    std = 0.2 * np.ones(n)
    return y_true, preds, std


@pytest.fixture
def multi_output_data():
    n = 20
    x = np.arange(n)
    y_true = np.stack([np.sin(x), np.cos(x)], axis=1)
    preds = y_true + 0.1 * np.random.randn(n, 2)
    std = 0.2 * np.ones((n, 2))
    return y_true, preds, std


def test_plot_confidence_interval_single_basic(single_output_data):
    y_true, preds, std = single_output_data
    # Should not raise error with default args
    plot_confidence_interval(y_true, preds, std)
    assert True


def test_plot_confidence_interval_single_errorbars(single_output_data):
    y_true, preds, std = single_output_data
    plot_confidence_interval(y_true, preds, std, plot_errorbars=True)
    assert True


def test_plot_confidence_interval_single_no_band(single_output_data):
    y_true, preds, std = single_output_data
    plot_confidence_interval(y_true, preds, std, plot_ci_band=False)
    assert True


def test_plot_confidence_interval_single_savefig(single_output_data):
    y_true, preds, std = single_output_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "plot.png")
        plot_confidence_interval(y_true, preds, std, savefig=path)
        assert os.path.isfile(path)


def test_plot_confidence_interval_multi_basic(multi_output_data):
    y_true, preds, std = multi_output_data
    plot_confidence_interval(y_true, preds, std)
    assert True


def test_plot_confidence_interval_multi_errorbars(multi_output_data):
    y_true, preds, std = multi_output_data
    plot_confidence_interval(y_true, preds, std, plot_errorbars=True)
    assert True


def test_plot_confidence_interval_multi_no_band(multi_output_data):
    y_true, preds, std = multi_output_data
    plot_confidence_interval(y_true, preds, std, plot_ci_band=False)
    assert True


def test_plot_confidence_interval_multi_savefig(multi_output_data):
    y_true, preds, std = multi_output_data
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "plot_multi.png")
        plot_confidence_interval(y_true, preds, std, savefig=path)
        assert os.path.isfile(path)


def test_plot_multioutput_confidence_intervals_runs():
    np.random.seed(0)
    X = np.random.randn(30, 3)
    Y = np.stack([X[:,0] + 0.1 * np.random.randn(30), X[:,1] - X[:,2] + 0.1 * np.random.randn(30)], axis=1)
    plot_multioutput_confidence_intervals(X, Y)
    assert True


def test_plot_multioutput_confidence_intervals_errorbars_and_band():
    np.random.seed(1)
    X = np.random.randn(20, 3)
    Y = np.stack([X[:,0] + 0.1 * np.random.randn(20), X[:,1] - X[:,2] + 0.1 * np.random.randn(20)], axis=1)
    plot_multioutput_confidence_intervals(X, Y, plot_errorbars=True, plot_ci_band=False)
    assert True


def test_plot_multioutput_confidence_intervals_savefig():
    np.random.seed(2)
    X = np.random.randn(15, 2)
    Y = np.stack([X[:,0] + 0.1 * np.random.randn(15), X[:,1] + 0.1 * np.random.randn(15)], axis=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "multiout.png")
        plot_multioutput_confidence_intervals(X, Y, savefig=path)
        assert os.path.isfile(path)


def test_plot_regressormixin_confidence_intervals_runs():
    np.random.seed(3)
    X = np.random.randn(20, 2)
    Y = np.stack([2*X[:,0] + 0.1 * np.random.randn(20), -X[:,1] + 0.1 * np.random.randn(20)], axis=1)
    plot_regressormixin_confidence_intervals(X, Y)
    assert True


def test_plot_regressormixin_confidence_intervals_errorbars_and_band():
    np.random.seed(4)
    X = np.random.randn(10, 3)
    Y = np.stack([X[:,0], X[:,1]], axis=1) + 0.1 * np.random.randn(10,2)
    plot_regressormixin_confidence_intervals(X, Y, plot_errorbars=True, plot_ci_band=False)
    assert True


def test_plot_regressormixin_confidence_intervals_savefig():
    np.random.seed(5)
    X = np.random.randn(12, 3)
    Y = np.stack([X[:,0], X[:,1]], axis=1) + 0.1 * np.random.randn(12,2)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "regmix.png")
        plot_regressormixin_confidence_intervals(X, Y, savefig=path)
        assert os.path.isfile(path)


def test_plot_multioutput_confidence_intervals_single_target():
    np.random.seed(123)
    n = 25
    X = np.random.randn(n, 4)
    # Single target (shape: n, 1)
    Y = 2 * X[:, 0].reshape(-1, 1) + 0.1 * np.random.randn(n, 1)
    plot_multioutput_confidence_intervals(X, Y)
    plt.close('all')
    assert True


def _make_fake_preds(n_samples, n_targets=1):
    y_true = np.random.randn(n_samples, n_targets)
    y_pred = y_true + np.random.randn(n_samples, n_targets) * 0.2
    y_std = np.abs(np.random.randn(n_samples, n_targets) * 0.5 + 0.1)
    if n_targets == 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_std = y_std.flatten()
    return y_pred, y_std, y_true


def test_plot_intervals_ordered_runs():
    y_pred, y_std, y_true = _make_fake_preds(30)
    pred_line, obs_line, ax = plot_intervals_ordered(y_pred, y_std, y_true)
    assert hasattr(pred_line, "get_xydata")
    assert hasattr(obs_line, "get_xydata")
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize("n_targets", [1, 2, 3, 4])
def test_plot_intervals_ordered_multi_axes_count(tmp_path, n_targets):
    n_samples = 24
    y_pred, y_std, y_true = _make_fake_preds(n_samples, n_targets)
    out_file = tmp_path / f"targets{n_targets}.png"
    # Should run and save file for n_targets
    plot_intervals_ordered_multi(y_pred, y_std, y_true, savefig=str(out_file))
    assert out_file.exists()


def test_plot_intervals_ordered_multi_grid_and_titles(tmp_path):
    n_samples, n_targets = 12, 5
    y_pred, y_std, y_true = _make_fake_preds(n_samples, n_targets)
    target_list = [f"MyTarget_{i}" for i in range(n_targets)]
    out_file = tmp_path / "ititles.png"
    plot_intervals_ordered_multi(
        y_pred, y_std, y_true, savefig=str(out_file), target_list=target_list, max_cols=2
    )
    # Just check the file is saved and can be loaded as a figure
    assert out_file.exists()
    fig = plt.imread(str(out_file))
    assert fig is not None


def test_plot_intervals_ordered_subset():
    y_pred, y_std, y_true = _make_fake_preds(50)
    # Should work for a subset (e.g., 10 points)
    _, _, ax = plot_intervals_ordered(y_pred, y_std, y_true, n_subset=10)
    assert isinstance(ax, plt.Axes)


def test_plot_intervals_ordered_multi_shares_y(tmp_path):
    n_samples, n_targets = 25, 3
    y_pred, y_std, y_true = _make_fake_preds(n_samples, n_targets)
    out_file = tmp_path / "sharey.png"
    plot_intervals_ordered_multi(y_pred, y_std, y_true, savefig=str(out_file))
    assert out_file.exists()


def test_plot_intervals_ordered_multi_handles_transpose(tmp_path):
    n_samples, n_targets = 18, 2
    y_pred, y_std, y_true = _make_fake_preds(n_samples, n_targets)
    # Test with transposed shape (n_targets, n_samples)
    out_file = tmp_path / "trans.png"
    plot_intervals_ordered_multi(y_pred.T, y_std.T, y_true.T, savefig=str(out_file))
    assert out_file.exists()


def test_plot_intervals_ordered_multi_custom_labels(tmp_path):
    n_samples, n_targets = 14, 2
    y_pred, y_std, y_true = _make_fake_preds(n_samples, n_targets)
    out_file = tmp_path / "customlbl.png"
    plot_intervals_ordered_multi(
        y_pred, y_std, y_true,
        suptitle="SuperTitle",
        supxlabel="XLabel",
        supylabel="YLabel",
        savefig=str(out_file)
    )
    assert out_file.exists()
