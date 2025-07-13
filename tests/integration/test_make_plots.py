# Copyright (c) 2025 takotime808

import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "make_plots.py"


def test_make_plots_script(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0
    assert (tmp_path / "metrics.png").exists()