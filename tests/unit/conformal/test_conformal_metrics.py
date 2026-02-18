# Copyright (c) 2025 takotime808

import numpy as np
import pytest

from multioutreg.conformal.metrics import (
    conformal_coverage,
    conformal_interval_width,
    conformal_summary,
    conditional_coverage,
)


class TestConformalCoverage:
    def test_perfect_coverage(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        df = conformal_coverage(y_true, y_lower, y_upper)
        assert df["coverage"].iloc[0] == 1.0

    def test_zero_coverage(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_lower = np.array([0.0, 0.0, 0.0])
        y_upper = np.array([1.0, 1.0, 1.0])
        df = conformal_coverage(y_true, y_lower, y_upper)
        assert df["coverage"].iloc[0] == 0.0

    def test_multi_output(self):
        y_true = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
        y_lower = np.array([[0, 5], [1, 15], [2, 25], [3, 35]])
        y_upper = np.array([[2, 15], [3, 25], [4, 35], [5, 45]])
        df = conformal_coverage(y_true, y_lower, y_upper)
        assert len(df) == 2
        assert df["coverage"].iloc[0] == 1.0
        assert df["coverage"].iloc[1] == 1.0

    def test_custom_output_names(self):
        y_true = np.array([[1], [2]])
        y_lower = np.array([[0], [1]])
        y_upper = np.array([[2], [3]])
        df = conformal_coverage(y_true, y_lower, y_upper, output_names=["temp"])
        assert df["output"].iloc[0] == "temp"


class TestConformalIntervalWidth:
    def test_constant_width(self):
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([2.0, 3.0, 4.0])
        df = conformal_interval_width(y_lower, y_upper)
        assert df["mean_width"].iloc[0] == 2.0
        assert df["std_width"].iloc[0] == 0.0

    def test_multi_output(self):
        y_lower = np.array([[0, 0], [1, 10]])
        y_upper = np.array([[2, 5], [3, 15]])
        df = conformal_interval_width(y_lower, y_upper)
        assert len(df) == 2


class TestConformalSummary:
    def test_summary_columns(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        df = conformal_summary(y_true, y_lower, y_upper, alpha=0.1)
        assert "nominal_coverage" in df.columns
        assert "coverage" in df.columns
        assert "coverage_gap" in df.columns
        assert "mean_width" in df.columns
        assert df["nominal_coverage"].iloc[0] == 0.9


class TestConditionalCoverage:
    def test_returns_correct_shape(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(100)
        y_lower = y_true - 2
        y_upper = y_true + 2
        centers, coverages = conditional_coverage(y_true, y_lower, y_upper, n_bins=5)
        assert len(centers) == 5
        assert len(coverages) == 5

    def test_perfect_conditional_coverage(self):
        y_true = np.linspace(0, 10, 100)
        y_lower = y_true - 1
        y_upper = y_true + 1
        centers, coverages = conditional_coverage(y_true, y_lower, y_upper, n_bins=5)
        assert np.all(coverages == 1.0)
