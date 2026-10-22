# Copyright (c) 2026 takotime808
"""Shared pytest configuration."""

import matplotlib

# Non-interactive backend: no plot windows, and plt.show() never blocks the run.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every matplotlib figure after each test."""
    yield
    plt.close("all")
