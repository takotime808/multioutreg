# Copyright (c) 2026 takotime808

import io
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from multioutreg import persistence
from multioutreg.persistence import ModelBundle, load_model, save_model
from multioutreg.surrogates import (
    BayesianRidgeSurrogate,
    LinearRegressionSurrogate,
    MultiFidelitySurrogate,
    PCAOutputSurrogate,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    Y = pd.DataFrame({"y1": X["a"] + X["b"], "y2": X["b"] - X["c"]})
    return X, Y


def test_round_trip_keeps_predictions_and_names(tmp_path, data):
    X, Y = data
    model = BayesianRidgeSurrogate().fit(X.to_numpy(), Y.to_numpy())
    path = tmp_path / "m.joblib"
    save_model(model, path, input_columns=X.columns, output_columns=Y.columns, metadata={"source": "unit"})
    bundle = load_model(path)

    assert isinstance(bundle, ModelBundle)
    assert bundle.input_columns == ["a", "b", "c"]
    assert bundle.output_columns == ["y1", "y2"]
    assert bundle.metadata["source"] == "unit" and bundle.metadata["saved_at"]
    assert bundle.versions["multioutreg"]
    mean, std = bundle.predict(X.to_numpy(), return_std=True)
    exp_mean, exp_std = model.predict(X.to_numpy(), return_std=True)
    np.testing.assert_allclose(mean, exp_mean)
    np.testing.assert_allclose(std, exp_std)


def test_dataframe_input_is_reordered_and_frames_returned(tmp_path, data):
    X, Y = data
    model = LinearRegressionSurrogate().fit(X.to_numpy(), Y.to_numpy())
    save_model(model, tmp_path / "m.joblib", input_columns=X.columns, output_columns=Y.columns)
    bundle = load_model(tmp_path / "m.joblib")
    shuffled = X[["c", "a", "b"]].assign(extra=1.0)
    pred = bundle.predict(shuffled, as_frame=True)
    assert list(pred.columns) == ["y1", "y2"]
    np.testing.assert_allclose(pred.to_numpy(), model.predict(X.to_numpy()))
    mean, std = bundle.predict(X, as_frame=True, return_std=True)
    assert list(std.columns) == ["y1", "y2"]
    with pytest.raises(KeyError, match="'b'"):
        bundle.predict(X.drop(columns="b"))


def test_fidelity_levels_inferred_from_model(tmp_path, data):
    X, Y = data
    mf = MultiFidelitySurrogate(LinearRegressionSurrogate, ["lo", "hi"]).fit(
        {"lo": (X.to_numpy(), Y.to_numpy()), "hi": (X.to_numpy(), Y.to_numpy())}
    )
    save_model(mf, tmp_path / "mf.joblib")
    bundle = load_model(tmp_path / "mf.joblib")
    assert bundle.fidelity_levels == ["lo", "hi"]
    np.testing.assert_allclose(bundle.predict(X.to_numpy()), mf.predict(X.to_numpy()))


def test_wrapper_surrogates_round_trip(tmp_path, data):
    X, Y = data
    model = PCAOutputSurrogate(BayesianRidgeSurrogate, n_components=2).fit(X.to_numpy(), Y.to_numpy())
    buf = io.BytesIO()
    save_model(model, buf)
    buf.seek(0)
    _, cov = load_model(buf).predict(X.to_numpy()[:3], return_cov=True)
    np.testing.assert_allclose(cov, model.predict(X.to_numpy()[:3], return_cov=True)[1])


def test_warns_on_version_mismatch(tmp_path, data, monkeypatch):
    X, Y = data
    save_model(LinearRegressionSurrogate().fit(X.to_numpy(), Y.to_numpy()), tmp_path / "m.joblib")
    real = persistence.package_versions()
    monkeypatch.setattr(persistence, "package_versions", lambda: {**real, "scikit-learn": "0.0.0"})
    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(tmp_path / "m.joblib")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_model(tmp_path / "m.joblib", check_versions=False)


def test_rejects_newer_format(tmp_path):
    joblib.dump({"format": persistence.FORMAT, "format_version": 99, "model": None}, tmp_path / "f.joblib")
    with pytest.raises(ValueError, match="newer"):
        load_model(tmp_path / "f.joblib")


def test_opens_grid_search_app_bundles_and_plain_dumps(tmp_path, data):
    X, Y = data
    model = LinearRegressionSurrogate().fit(X.to_numpy(), Y.to_numpy())
    app_bundle = {
        "pipeline": model,
        "input_columns": ["a", "b", "c"],
        "output_columns": ["y1", "y2"],
        "conformal": {"alpha": 0.1, "quantiles": {"y1": 0.5, "y2": 0.4}},
        "config": {"models": []},
        "versions": persistence.package_versions(),
    }
    joblib.dump(app_bundle, tmp_path / "app.joblib")
    b = load_model(tmp_path / "app.joblib")
    assert b.output_columns == ["y1", "y2"] and b.conformal["alpha"] == 0.1
    np.testing.assert_allclose(b.predict(X), model.predict(X.to_numpy()))

    joblib.dump(model, tmp_path / "raw.joblib")
    raw = load_model(tmp_path / "raw.joblib")
    assert raw.input_columns is None and raw.versions == {}
    np.testing.assert_allclose(raw.predict(X.to_numpy()), model.predict(X.to_numpy()))
