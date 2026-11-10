# Copyright (c) 2026 takotime808
"""The core install must not need the ``gui``, ``torch`` or ``ts`` extras.

Each check runs in a fresh interpreter with those packages made unimportable,
so it holds even though the test environment has every extra installed.
"""

import subprocess
import sys

import pytest

# Top-level modules provided only by the gui / torch / ts extras.
OPTIONAL = ("torch", "chronos", "streamlit", "plotly", "pyarrow", "gpytorch", "boto3", "dotenv")

_BLOCKER = f"""
import importlib.abc, sys

class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {OPTIONAL!r}:
            raise ImportError("blocked optional dependency: " + name)

sys.meta_path.insert(0, _Block())
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + code],
        capture_output=True, text=True, timeout=300,
    )


@pytest.mark.parametrize("module", [
    "multioutreg",
    "multioutreg.active",
    "multioutreg.conformal",
    "multioutreg.figures",
    "multioutreg.metrics",
    "multioutreg.model_selection",
    "multioutreg.performance",
    "multioutreg.report.build",
    "multioutreg.report.render",
    "multioutreg.sampling",
    "multioutreg.surrogates",
    "multioutreg.time_series.chronos_adapter",
    "multioutreg.time_series.cv",
    "multioutreg.time_series.lag_forecaster",
    "mor_cli",
    "mor_cli.grid_search",
    "mor_cli.grid_search_auto_detect",
    "mor_cli.detect_sampling",
    "mor_cli.ts_forecast",
    "mor_cli.ts_pipeline",
])
def test_imports_without_optional_extras(module):
    result = _run(f"import {module}")
    assert result.returncode == 0, result.stderr[-2000:]


def test_chronos_forecaster_names_the_extra():
    result = _run(
        "from multioutreg.time_series.chronos_adapter import ChronosForecaster\n"
        "try:\n"
        "    ChronosForecaster()\n"
        "except ImportError as e:\n"
        "    assert 'multioutreg[ts]' in str(e), e\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_torch_surrogate_names_the_extra():
    result = _run(
        "try:\n"
        "    import multioutreg.surrogates.bnn_network\n"
        "except ImportError as e:\n"
        "    assert 'multioutreg[torch]' in str(e), e\n"
        "else:\n"
        "    raise SystemExit('expected ImportError')\n"
    )
    assert result.returncode == 0, result.stderr[-2000:]
