# Copyright (c) 2025 takotime808

import runpy
from pathlib import Path

import matplotlib.image as mpimg

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "make_plots.py"


def test_make_plots_script(tmp_path, monkeypatch):
    # In-process (not a subprocess) so the run reuses this session's imports;
    # an exception in the script fails the test with its own traceback.
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    out = tmp_path / "metrics.png"
    assert out.exists()
    img = mpimg.imread(out)
    assert img.shape[0] > 100 and img.shape[1] > 100
