# Copyright (c) 2026 takotime808

"""Background jobs: inline mode, a real spawn pool, and the inline fallback."""

import base64
import sys
import types

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
    background._discard("plots")  # fresh workers, started by this test
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


def test_workers_do_not_rerun_the_main_script(real_pool, tmp_path, monkeypatch):
    # Streamlit registers the page script as __main__; a spawned worker would
    # normally re-import it, i.e. run the whole page again.
    marker = tmp_path / "page_ran"
    script = tmp_path / "page.py"
    script.write_text(f"open({str(marker)!r}, 'w').close()\n")
    fake_main = types.ModuleType("__main__")
    fake_main.__file__ = str(script)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    assert background.submit("plots", sum, [1, 2, 3]).result() == 6
    assert not marker.exists()
    assert sys.modules["__main__"] is fake_main  # restored after the submit
