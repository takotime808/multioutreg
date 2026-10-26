# Copyright (c) 2026 takotime808

"""Background jobs: inline mode, a real spawn pool, and the inline fallback."""

import base64

import numpy as np
import pandas as pd
import pytest

from multioutreg.figures.doe_plots import pair_plot_b64
from multioutreg.gui import background


def _boom():
    raise ValueError("boom")


def test_disabled_runs_inline(monkeypatch):
    monkeypatch.setenv("MOR_BACKGROUND_JOBS", "0")
    calls = []
    job = background.submit("plots", lambda x: calls.append(x) or x * 2, 21)
    assert job.done()
    assert calls == []  # nothing runs until the result is asked for
    assert job.result() == 42 and job.result() == 42
    assert calls == [21]


def test_inline_errors_propagate(monkeypatch):
    monkeypatch.setenv("MOR_BACKGROUND_JOBS", "0")
    with pytest.raises(ValueError, match="boom"):
        background.submit("plots", _boom).result()


@pytest.fixture
def real_pool(monkeypatch):
    monkeypatch.setenv("MOR_BACKGROUND_JOBS", "1")
    yield
    background._discard("plots")


def test_pair_plot_runs_in_a_worker_process(real_pool):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(size=40), "y": rng.normal(size=40)})
    job = background.submit("plots", pair_plot_b64, df, ["x"], ["y"])
    assert job._future is not None  # really submitted, not inline
    result = job.result()
    assert base64.b64decode(result["img_b64"]).startswith(b"\x89PNG")
    assert job.done()


def test_unpicklable_job_falls_back_to_inline(real_pool):
    job = background.submit("plots", lambda: "ran inline")  # lambdas don't pickle
    assert job.result() == "ran inline"


def test_worker_error_is_raised_by_result(real_pool):
    job = background.submit("plots", _boom)
    with pytest.raises(ValueError, match="boom"):
        job.result()
