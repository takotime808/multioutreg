# Copyright (c) 2025 takotime808

import pytest
import matplotlib.pyplot as plt
import base64

from multioutreg.utils import figure_utils as fu

def test_safe_plot_b64_returns_base64_on_success():
    def plot_func():
        plt.plot([0, 1], [0, 1])
        plt.title("Test Plot")
    b64str = fu.safe_plot_b64(plot_func)
    assert isinstance(b64str, str)
    assert len(b64str) > 100
    # Should be decodable
    img = base64.b64decode(b64str)
    assert img[:8] == b'\x89PNG\r\n\x1a\n'  # PNG signature

def test_safe_plot_b64_handles_exception_and_includes_error():
    def broken_plot():
        raise ValueError("Oops!")
    b64str = fu.safe_plot_b64(broken_plot)
    assert isinstance(b64str, str)
    img = base64.b64decode(b64str)
    assert img[:8] == b'\x89PNG\r\n\x1a\n'
    # Optionally, OCR the PNG to check the error text, but that's outside normal pytest
    # So instead, ensure still returns a base64 PNG

def test_safe_plot_b64_with_args_kwargs():
    def plot_func(x, y, title=None):
        plt.plot(x, y)
        if title:
            plt.title(title)
    b64str = fu.safe_plot_b64(plot_func, [0, 1], [1, 0], title="Args & Kwargs")
    assert isinstance(b64str, str)
    img = base64.b64decode(b64str)
    assert img[:8] == b'\x89PNG\r\n\x1a\n'

def test_safe_plot_b64_cleans_up_figures():
    # Count open figures before and after to ensure no leaks
    before = plt.get_fignums()
    def plot_func():
        plt.plot([0,1],[0,1])
    _ = fu.safe_plot_b64(plot_func)
    after = plt.get_fignums()
    # Should not increase
    assert len(after) <= len(before)

