# Copyright (c) 2026 takotime808

"""``multioutreg ts-forecast`` with a fake Chronos model (the real one downloads weights)."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mor_cli import ts_forecast as cli
from multioutreg.time_series.chronos_adapter import ForecastResult


class FakeChronos:
    """Forecasts ``100 * series_index + quantile`` at every step."""

    instances = []

    def __init__(self, model_name):
        self.model_name = model_name
        FakeChronos.instances.append(self)

    def fit(self, series):
        self.series = series
        return self

    def predict(self, prediction_length, quantiles):
        n = len(self.series)
        q = np.array(quantiles)
        values = 100.0 * np.arange(n)[:, None, None] + q[None, :, None]
        values = np.broadcast_to(values, (n, len(q), prediction_length)).copy()
        return ForecastResult(quantiles=values, q_levels=tuple(quantiles), ids=tuple(self.series))


@pytest.fixture
def csv(tmp_path, monkeypatch):
    FakeChronos.instances.clear()
    monkeypatch.setattr(cli, "ChronosForecaster", FakeChronos)
    path = tmp_path / "series.csv"
    pd.DataFrame({
        "t": np.arange(10),
        "a": np.arange(10, dtype=float),
        "b": np.r_[np.arange(9, dtype=float), np.nan],
        "label": list("xxxxxyyyyy"),
    }).to_csv(path, index=False)
    return path


def _invoke(*args):
    return CliRunner().invoke(cli.app, [*map(str, args)])


def test_writes_tidy_long_csv(csv, tmp_path):
    out = tmp_path / "fc.csv"
    result = _invoke(csv, "--value-cols", "a,b", "--horizon", 3,
                     "--quantiles", "0.1,0.9", "--out", out, "--model", "m")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"written": str(out), "ids": ["a", "b"], "quantiles": [0.1, 0.9]}

    fake = FakeChronos.instances[-1]
    assert fake.model_name == "m"
    assert len(fake.series["b"]) == 9  # NaNs dropped before fitting

    df = pd.read_csv(out)
    assert list(df.columns) == ["id", "step", "quantile", "value"]
    assert len(df) == 2 * 2 * 3
    assert sorted(df["step"].unique()) == [1, 2, 3]
    row = df[(df.id == "b") & (df["quantile"] == 0.9) & (df.step == 2)]
    assert row["value"].item() == pytest.approx(100.9)


def test_defaults_to_first_numeric_column_other_than_time(csv, tmp_path):
    out = tmp_path / "fc.csv"
    result = _invoke(csv, "--time-col", "t", "--horizon", 2, "--out", out)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ids"] == ["a"]
    assert set(pd.read_csv(out)["quantile"]) == {0.1, 0.5, 0.9}


@pytest.mark.parametrize("args, message", [
    (["--value-cols", "a,zzz"], "Missing columns"),
    (["--value-cols", "a", "--quantiles", "0.5,1.5"], "Quantiles must be in (0,1)"),
    (["--value-cols", "a", "--quantiles", " , "], "at least one quantile"),
])
def test_bad_options_are_rejected(csv, tmp_path, args, message):
    result = _invoke(csv, "--horizon", 2, "--out", tmp_path / "fc.csv", *args)
    assert result.exit_code != 0
    assert message in str(result.exception) + result.output
    assert not (tmp_path / "fc.csv").exists()


def test_no_numeric_columns_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ChronosForecaster", FakeChronos)
    path = tmp_path / "text.csv"
    pd.DataFrame({"label": ["x", "y"]}).to_csv(path, index=False)
    result = _invoke(path, "--horizon", 1, "--out", tmp_path / "fc.csv")
    assert result.exit_code != 0
    assert "No numeric columns" in str(result.exception) + result.output
