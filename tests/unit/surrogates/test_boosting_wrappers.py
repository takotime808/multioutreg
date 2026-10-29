# Copyright (c) 2026 takotime808

"""LightGBM / XGBoost wrappers: missing-dependency guard and set_params."""

import importlib

import numpy as np
import pytest

WRAPPERS = [
    pytest.param("lightgbm_sklearn", "LightGBMSurrogate", "_LIGHTGBM_AVAILABLE", id="lightgbm"),
    pytest.param("xgboost_sklearn", "XGBoostSurrogate", "_XGBOOST_AVAILABLE", id="xgboost"),
]


def _load(module, cls_name, flag):
    mod = importlib.import_module(f"multioutreg.surrogates.{module}")
    if not getattr(mod, flag):
        pytest.skip(f"{module}: optional dependency not installed")
    return mod, getattr(mod, cls_name)


@pytest.mark.parametrize("module, cls_name, flag", WRAPPERS)
def test_missing_dependency_raises_a_helpful_import_error(monkeypatch, module, cls_name, flag):
    mod = importlib.import_module(f"multioutreg.surrogates.{module}")
    model = getattr(mod, cls_name)(n_estimators=5) if getattr(mod, flag) else None
    monkeypatch.setattr(mod, flag, False)
    with pytest.raises(ImportError, match="pip install"):
        getattr(mod, cls_name)()
    if model is not None:
        with pytest.raises(ImportError, match="pip install"):
            model.set_params(n_estimators=10)


@pytest.mark.parametrize("module, cls_name, flag", WRAPPERS)
def test_set_params_updates_named_and_extra_kwargs(module, cls_name, flag):
    _, cls = _load(module, cls_name, flag)
    model = cls(n_estimators=5, learning_rate=0.1, random_state=0)
    assert model.set_params(n_estimators=7, random_state=3) is model

    params = model.get_params()
    assert params["n_estimators"] == 7
    assert params["random_state"] == 3
    est = model.model.estimator
    assert est.get_params()["n_estimators"] == 7
    assert est.get_params()["random_state"] == 3

    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40, 2))
    Y = np.column_stack([X[:, 0], X[:, 1] ** 2])
    assert model.fit(X, Y).predict(X).shape == Y.shape
    assert all(e.get_params()["n_estimators"] == 7 for e in model.model.estimators_)
