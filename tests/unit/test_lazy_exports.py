# Copyright (c) 2026 takotime808

"""Lazily resolved package exports (``multioutreg`` and ``multioutreg.time_series``)."""

import importlib

import pytest

import multioutreg
import multioutreg.time_series as ts

# Optional backends whose modules may need extra packages to import.
OPTIONAL = {"ProphetForecaster", "NeuralForecaster"}


@pytest.mark.parametrize("pkg", [multioutreg, ts], ids=["multioutreg", "time_series"])
def test_unknown_attribute_raises_attribute_error(pkg):
    with pytest.raises(AttributeError, match="no_such_name"):
        pkg.no_such_name
    assert not hasattr(pkg, "no_such_name")


@pytest.mark.parametrize("pkg", [multioutreg, ts], ids=["multioutreg", "time_series"])
def test_dir_lists_every_export(pkg):
    assert set(pkg.__all__) <= set(dir(pkg))


@pytest.mark.parametrize("name", sorted(set(ts.__all__) - OPTIONAL))
def test_time_series_exports_resolve_and_are_cached(monkeypatch, name):
    monkeypatch.delitem(vars(ts), name, raising=False)
    value = getattr(ts, name)
    submodule = ts._EXPORTS[name]
    expected = (importlib.import_module(f"multioutreg.time_series.{name}") if submodule is None
                else getattr(importlib.import_module(f"multioutreg.time_series.{submodule}"), name))
    assert value is expected
    assert vars(ts)[name] is value  # later lookups skip __getattr__


@pytest.mark.parametrize("name", sorted(multioutreg._TS_EXPORTS))
def test_top_level_time_series_exports_match_the_subpackage(monkeypatch, name):
    monkeypatch.delitem(vars(multioutreg), name, raising=False)
    assert getattr(multioutreg, name) is getattr(ts, multioutreg._TS_EXPORTS[name])
    assert name in vars(multioutreg)
