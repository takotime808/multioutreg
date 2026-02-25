# Copyright (c) 2025 takotime808

import types
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a mock neuralforecast module
# ---------------------------------------------------------------------------

def _make_mock_nf_module():
    """Return a minimal mock of the neuralforecast package."""
    nf_mod = types.ModuleType("neuralforecast")
    models_mod = types.ModuleType("neuralforecast.models")

    class FakeModel:
        def __init__(self, h, input_size, max_steps, **kwargs):
            self.h = h
            self.__class__.__name__ = self.__class__.__name__  # preserve class name

    class FakeNBEATS(FakeModel):
        pass

    class FakeNHITS(FakeModel):
        pass

    class FakeNeuralForecast:
        def __init__(self, models, freq):
            self.models = models
            self.freq = freq
            self._h = models[0].h if models else 1

        def fit(self, df, val_size=None, verbose=False):
            self._val_size = val_size
            return self

        def predict(self, level=None):
            h = self._h
            model_name = self.models[0].__class__.__name__
            data = {
                "unique_id": ["y"] * h,
                "ds": pd.date_range("2023-06-01", periods=h, freq="D"),
                model_name: np.ones(h) * 5.0,
            }
            if level:
                lvl = level[0]
                data[f"{model_name}-lo-{lvl}"] = np.ones(h) * 3.0
                data[f"{model_name}-hi-{lvl}"] = np.ones(h) * 7.0
            return pd.DataFrame(data)

    nf_mod.NeuralForecast = FakeNeuralForecast
    models_mod.NBEATS = FakeNBEATS
    models_mod.NHITS = FakeNHITS
    return nf_mod, models_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNeuralForecasterImportGuard:

    def test_raises_if_neuralforecast_not_available(self):
        """NeuralForecaster.__init__ should raise ImportError when neuralforecast is absent."""
        import multioutreg.time_series.neuralforecast_adapter as nfa_mod
        orig = nfa_mod._NEURALFORECAST_AVAILABLE
        orig_nf = nfa_mod._NeuralForecast
        orig_nbeats = nfa_mod._NBEATS
        orig_nhits = nfa_mod._NHITS
        try:
            nfa_mod._NEURALFORECAST_AVAILABLE = False
            nfa_mod._NeuralForecast = None
            nfa_mod._NBEATS = None
            nfa_mod._NHITS = None
            with pytest.raises(ImportError, match="neuralforecast is required"):
                nfa_mod.NeuralForecaster()
        finally:
            nfa_mod._NEURALFORECAST_AVAILABLE = orig
            nfa_mod._NeuralForecast = orig_nf
            nfa_mod._NBEATS = orig_nbeats
            nfa_mod._NHITS = orig_nhits

    def test_invalid_model_type_raises(self):
        import multioutreg.time_series.neuralforecast_adapter as nfa_mod
        nf_mod, models_mod = _make_mock_nf_module()
        orig_avail = nfa_mod._NEURALFORECAST_AVAILABLE
        orig_nf = nfa_mod._NeuralForecast
        orig_nbeats = nfa_mod._NBEATS
        orig_nhits = nfa_mod._NHITS
        try:
            nfa_mod._NEURALFORECAST_AVAILABLE = True
            nfa_mod._NeuralForecast = nf_mod.NeuralForecast
            nfa_mod._NBEATS = models_mod.NBEATS
            nfa_mod._NHITS = models_mod.NHITS
            with pytest.raises(ValueError, match="model_type"):
                nfa_mod.NeuralForecaster(model_type="transformer")
        finally:
            nfa_mod._NEURALFORECAST_AVAILABLE = orig_avail
            nfa_mod._NeuralForecast = orig_nf
            nfa_mod._NBEATS = orig_nbeats
            nfa_mod._NHITS = orig_nhits


class TestNeuralForecasterWithMock:

    @pytest.fixture(autouse=True)
    def patch_nf(self):
        """Inject fake neuralforecast module so no real installation is needed."""
        nf_mod, models_mod = _make_mock_nf_module()
        import multioutreg.time_series.neuralforecast_adapter as nfa_mod
        self._nfa_mod = nfa_mod
        orig_avail = nfa_mod._NEURALFORECAST_AVAILABLE
        orig_nf = nfa_mod._NeuralForecast
        orig_nbeats = nfa_mod._NBEATS
        orig_nhits = nfa_mod._NHITS
        nfa_mod._NEURALFORECAST_AVAILABLE = True
        nfa_mod._NeuralForecast = nf_mod.NeuralForecast
        nfa_mod._NBEATS = models_mod.NBEATS
        nfa_mod._NHITS = models_mod.NHITS
        yield
        nfa_mod._NEURALFORECAST_AVAILABLE = orig_avail
        nfa_mod._NeuralForecast = orig_nf
        nfa_mod._NBEATS = orig_nbeats
        nfa_mod._NHITS = orig_nhits

    def _make_series(self, n=200):
        dates = pd.date_range("2022-01-01", periods=n, freq="D")
        vals = np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.default_rng(0).normal(0, 0.1, n)
        return pd.Series(vals, index=dates)

    def test_fit_returns_self(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        assert f.fit(self._make_series()) is f

    def test_predict_returns_forecast_result(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        from multioutreg.time_series.chronos_adapter import ForecastResult
        f = NeuralForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=8)
        assert isinstance(res, ForecastResult)

    def test_nbeats_shape(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster(model_type="nbeats")
        f.fit(self._make_series(200))
        res = f.predict(prediction_length=10, quantiles=(0.1, 0.5, 0.9))
        assert res.quantiles.shape == (1, 3, 10)

    def test_nhits_shape(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster(model_type="nhits")
        f.fit(self._make_series(200))
        res = f.predict(prediction_length=7, quantiles=(0.1, 0.5, 0.9))
        assert res.quantiles.shape == (1, 3, 7)

    def test_q_levels_stored(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=5, quantiles=(0.1, 0.5, 0.9))
        assert list(res.q_levels) == [0.1, 0.5, 0.9]

    def test_quantile_ordering(self):
        """lower ≤ median ≤ upper at every step."""
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=8, quantiles=(0.1, 0.5, 0.9))
        q = res.quantiles[0]  # (3, 8)
        assert np.all(q[0] <= q[1] + 1e-9), "lower > median at some step"
        assert np.all(q[1] <= q[2] + 1e-9), "median > upper at some step"

    def test_fit_with_numpy_array(self):
        """Passing a plain np.ndarray should not raise."""
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        y = np.sin(np.linspace(0, 4 * np.pi, 100))
        f.fit(y, freq="D")
        res = f.predict(prediction_length=5)
        assert res.quantiles.shape == (1, 3, 5)

    def test_predict_before_fit_raises(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            f.predict(prediction_length=5)

    def test_ids_field(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=5)
        assert res.ids == ("y",)

    def test_custom_quantiles(self):
        from multioutreg.time_series.neuralforecast_adapter import NeuralForecaster
        f = NeuralForecaster()
        f.fit(self._make_series())
        res = f.predict(prediction_length=6, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95))
        assert res.quantiles.shape == (1, 5, 6)
