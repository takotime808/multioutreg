# Copyright (c) 2025 takotime808
"""Plan §2.4a: ts_dynamic_fit imports cleanly and writes no files."""

import json
import logging
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch


def test_package_import_is_lazy():
    code = (
        "import sys, multioutreg.time_series.ts_dynamic_fit as t;"
        "print('statsmodels' in sys.modules, 'multioutreg.time_series.ts_dynamic_fit.algs.lstm' in sys.modules);"
        "t.ARIMA;"
        "print('multioutreg.time_series.ts_dynamic_fit.algs.arima' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    lines = out.stdout.strip().splitlines()
    assert lines[-2] == "False False"
    assert lines[-1] == "True"


def test_package_exports():
    import multioutreg.time_series.ts_dynamic_fit as tdf

    for name in tdf.__all__:
        assert getattr(tdf, name) is not None
    with pytest.raises(AttributeError):
        tdf.does_not_exist


def test_import_does_not_configure_root_logger():
    code = (
        "import logging;"
        "from multioutreg.time_series.ts_dynamic_fit import ARIMA, SARIMA, LSTM, DataProcessor, Ranker;"
        "print(len(logging.getLogger().handlers))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "0"


@pytest.fixture(scope="module")
def fitted_lstm(tmp_path_factory):
    from multioutreg.time_series.ts_dynamic_fit import LSTM

    torch.manual_seed(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"value": 10 + np.sin(np.arange(30) / 3.0)}, index=idx)
    workdir = tmp_path_factory.mktemp("lstm_run")
    mp = pytest.MonkeyPatch()
    mp.chdir(workdir)
    try:
        lstm = LSTM(df, fut_pred=10, feature_column="value", train_window=5)
        metadata, _, model = lstm.run()
    finally:
        mp.undo()
    return df, lstm, metadata, model, workdir


def test_lstm_keeps_scaler_and_writes_no_files(fitted_lstm):
    _, lstm, metadata, _, workdir = fitted_lstm
    assert hasattr(lstm.scaler_, "transform")
    assert list(workdir.iterdir()) == []
    json.dumps(metadata)  # main.py / ts-pipeline serialize this dict


def _registry(tmp_path, metadata, model):
    meta_path = tmp_path / "metadata.json"
    model_path = tmp_path / "LSTM.pt"
    meta_path.write_text(json.dumps(metadata))
    torch.save(model.state_dict(), model_path)
    return str(meta_path), str(model_path)


def test_infer_lstm_with_scaler_argument(fitted_lstm, tmp_path):
    from multioutreg.time_series.ts_dynamic_fit.src.infer import infer

    df, lstm, metadata, model, _ = fitted_lstm
    meta_path, model_path = _registry(tmp_path, metadata, model)
    preds = infer(df, "value", metadata_path=meta_path, model_path=model_path, scaler=lstm.scaler_)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


def test_infer_lstm_reads_scaler_next_to_model(fitted_lstm, tmp_path):
    from multioutreg.time_series.ts_dynamic_fit.src.infer import infer

    df, lstm, metadata, model, _ = fitted_lstm
    meta_path, model_path = _registry(tmp_path, metadata, model)
    joblib.dump(lstm.scaler_, tmp_path / "scaler.pkl")
    expected = infer(df, "value", metadata_path=meta_path, model_path=model_path, scaler=lstm.scaler_)
    preds = infer(df, "value", metadata_path=meta_path, model_path=model_path)
    np.testing.assert_allclose(preds, expected)


def test_infer_lstm_without_scaler_raises(fitted_lstm, tmp_path):
    from multioutreg.time_series.ts_dynamic_fit.src.infer import infer

    df, _, metadata, model, _ = fitted_lstm
    meta_path, model_path = _registry(tmp_path, metadata, model)
    with pytest.raises(FileNotFoundError, match="scaler"):
        infer(df, "value", metadata_path=meta_path, model_path=model_path)


def test_engineer_features_has_no_nans_and_no_deprecated_fillna(recwarn):
    from multioutreg.time_series.ts_dynamic_fit import DataProcessor

    idx = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame({"ds": idx, "y": np.arange(40, dtype=float)})
    proc = DataProcessor()
    proc.load_data(df)
    proc.preprocess_data(datetime_col="ds", target_col="y", freq="1D")
    out = proc.engineer_features(target_col="y")
    assert not out.isna().any().any()
    assert not [w for w in recwarn if "fillna" in str(w.message) and "method" in str(w.message)]


def test_log_messages_go_to_module_logger(caplog):
    from multioutreg.time_series.ts_dynamic_fit import ARIMA

    model = ARIMA(pd.DataFrame({"v": np.arange(10.0)}), feature_column="v")
    with caplog.at_level(logging.INFO, logger="multioutreg.time_series.ts_dynamic_fit"):
        model.log_to_file("hello")
    assert any(r.name.endswith("algs.arima") and r.message == "hello" for r in caplog.records)
