# Copyright (c) 2026 takotime808

"""Stacked Variable Fidelity Model surrogates.

Two classes are provided:

* StackedVFMSurrogate — Recursive feature-augmentation multi-fidelity model.
  Each fidelity level receives the original features augmented with all
  lower-level surrogate predictions (and optionally their stds), enabling
  nonlinear cross-fidelity information transfer with any existing surrogate.

* AdditiveCorrectionVFM — Two-level autoregressive correction (Kennedy-O'Hagan AR1).
  A per-output scale rho is fitted by least squares of Y_hi on f_lo(X_hi), a
  correction surrogate is trained on delta = Y_hi - rho * f_lo(X_hi), and the
  prediction is f_hi(x) = rho * f_lo(x) + delta(x).

References
----------
Perdikaris et al. (2017) "Nonlinear information fusion algorithms for
data-efficient multi-fidelity modelling." Proceedings of the Royal Society A.

Kennedy & O'Hagan (2000) "Predicting the output from a complex computer code
when fast approximations are available." Biometrika.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from multioutreg.surrogates.conformal_mixin import ConformalMixin

# Rows per surrogate call when propagating uncertainty by Monte Carlo.
_MC_MAX_ROWS = 50_000


def _as_2d(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _row_keys(X: np.ndarray) -> List[bytes]:
    return [row.tobytes() for row in np.ascontiguousarray(X, dtype=np.float64)]


class _OOFPredictor:
    """Out-of-fold predictions of a model at points of its own training set.

    A higher level is trained on a lower model's predictions at the higher
    level's inputs. When those inputs are also training points of the lower
    model (nested designs, the usual case), the full model's predictions there
    are in-sample and much more accurate than at new points, so the higher
    level learns to trust them too much. This class replaces those predictions
    with K-fold out-of-fold ones; query rows the model was not trained on keep
    the full model's prediction. Duplicate rows share a fold.

    Fold models are fitted lazily, only when a query row overlaps the training
    set.
    """

    def __init__(self, make_model, X_raw, X_fit, Y, cv, random_state):
        self._make_model = make_model
        self._X_fit = X_fit
        self._Y = Y
        keys = _row_keys(X_raw)
        uniq = list(dict.fromkeys(keys))
        self.n_splits = min(int(cv), len(uniq)) if cv else 0
        self._fold_of_key: Dict[bytes, int] = {}
        if self.n_splits >= 2:
            perm = np.random.default_rng(random_state).permutation(len(uniq))
            for f, idx in enumerate(np.array_split(perm, self.n_splits)):
                for j in idx:
                    self._fold_of_key[uniq[j]] = f
        self._train_folds = np.array([self._fold_of_key.get(k, -1) for k in keys])
        self._fold_models: Optional[list] = None

    def predict(self, full_model, X_raw_query, X_in_query, return_std):
        """Predict at ``X_in_query``, out-of-fold where rows overlap training."""
        if return_std:
            pred, std = full_model.predict(X_in_query, return_std=True)
            pred, std = _as_2d(pred).copy(), _as_2d(std).copy()
        else:
            pred, std = _as_2d(full_model.predict(X_in_query)).copy(), None

        if self.n_splits < 2:
            return pred, std
        q_folds = np.array(
            [self._fold_of_key.get(k, -1) for k in _row_keys(X_raw_query)]
        )
        if not (q_folds >= 0).any():
            return pred, std

        if self._fold_models is None:
            self._fold_models = []
            for f in range(self.n_splits):
                mask = self._train_folds != f
                m = self._make_model()
                m.fit(self._X_fit[mask], self._Y[mask])
                self._fold_models.append(m)

        for f in np.unique(q_folds[q_folds >= 0]):
            rows = q_folds == f
            m = self._fold_models[f]
            if return_std:
                p, s = m.predict(X_in_query[rows], return_std=True)
                std[rows] = _as_2d(s)
            else:
                p = m.predict(X_in_query[rows])
            pred[rows] = _as_2d(p)
        return pred, std


class StackedVFMSurrogate(ConformalMixin):
    """Stacked Variable Fidelity Model surrogate.

    Implements the recursive feature-augmentation approach described in
    Perdikaris et al. (2017).  At each fidelity level k, the input feature
    matrix is augmented with the predictions (and optionally predicted stds)
    of all lower-level surrogates before fitting, enabling the higher-fidelity
    surrogate to learn a nonlinear correction over the lower-fidelity
    prediction.

    Parameters
    ----------
    fidelity_levels : list[str]
        Ordered list of fidelity level names from lowest to highest.
        Must have at least 2 elements.
    surrogate_cls : type | list[type] | None
        Surrogate class(es) to instantiate per level.  A single class is
        used for all levels.  A list must have the same length as
        ``fidelity_levels`` and is used element-wise (heterogeneous experts).
        Defaults to ``RandomForestSurrogate``.
    surrogate_params : dict | list[dict] | None
        Constructor kwargs for each surrogate.  A single dict applies to all
        levels; a list is paired element-wise with ``surrogate_cls``.
    augment_with_std : bool, default False
        If True, the predicted standard deviation of level k is also appended
        to the feature matrix for level k+1 alongside the predictions.
        Requires each surrogate to support ``predict(X, return_std=True)``.

        Note: std features are treated as deterministic inputs, which is an
        approximation (see Perdikaris et al. 2017).  If the surrogate does not
        natively produce uncertainty (e.g. LinearRegressionSurrogate), the std
        columns will be all-zero and uninformative.
    output_dim_mismatch : {"error", "truncate", "pad"}, default "error"
        Policy when the augmented feature count at predict time does not match
        what was seen at fit time (can arise if surrogates are swapped via
        ``set_params`` without re-fitting).
        ``"error"`` raises ValueError.
        ``"truncate"`` clips extra augmentation columns.
        ``"pad"`` zero-pads missing augmentation columns.
    oof_cv : int or None, default 5
        Folds used for out-of-fold lower-level predictions while fitting.
        When a level's training inputs are also training inputs of a lower
        level (nested designs), the lower model's predictions there are
        in-sample and over-accurate; the higher level then over-trusts them
        and does worse at new points. Those rows instead get predictions from
        a lower model fitted without them. Rows that are not lower-level
        training points use the full model, as does ``predict``. ``None`` or
        ``0`` uses in-sample predictions (the old behaviour).
    random_state : int or None, default 0
        Seed for the out-of-fold split and the Monte Carlo draws.
    n_mc_samples : int, default 100
        Draws used by ``predict(..., return_std=True)`` to propagate
        lower-level uncertainty. Each lower level's prediction is replaced
        by a sample from ``N(mean, std)`` and passed up the stack; the
        returned std combines the top level's own std with the spread of its
        mean across draws. Without this the top level treats lower-level
        predictions as exact and its intervals are much too narrow. ``0``
        returns the top level's std alone (the old behaviour). Lower-level
        surrogates must support ``predict(X, return_std=True)``.

    Attributes
    ----------
    surrogates_ : dict[str, surrogate]
        Fitted surrogate per level, keyed by level name.
    n_outputs_per_level_ : dict[str, int]
        Number of outputs for each level as seen at fit time.
    augmented_n_features_per_level_ : dict[str, int]
        Total number of features (original + augmented) used for each level.
    n_features_in_ : int
        Number of raw input features (from level 0 data).

    Examples
    --------
    >>> import numpy as np
    >>> from multioutreg.surrogates import StackedVFMSurrogate
    >>> from multioutreg.surrogates import GaussianProcessSurrogate, RandomForestSurrogate
    >>> rng = np.random.default_rng(0)
    >>> X_lo = rng.standard_normal((80, 3))
    >>> Y_lo = np.sin(X_lo).sum(axis=1, keepdims=True)
    >>> X_hi = rng.standard_normal((30, 3))
    >>> Y_hi = np.sin(X_hi).sum(axis=1, keepdims=True) + 0.1 * rng.standard_normal((30, 1))
    >>> model = StackedVFMSurrogate(
    ...     fidelity_levels=["lo", "hi"],
    ...     surrogate_cls=[RandomForestSurrogate, GaussianProcessSurrogate],
    ... )
    >>> model.fit({"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)})
    StackedVFMSurrogate(...)
    >>> y_pred = model.predict(X_hi)
    >>> y_pred.shape
    (30, 1)
    """

    _multi_output = True

    def __init__(
        self,
        fidelity_levels: List[str],
        surrogate_cls=None,
        surrogate_params: Union[dict, List[dict], None] = None,
        augment_with_std: bool = False,
        output_dim_mismatch: Literal["error", "truncate", "pad"] = "error",
        oof_cv: Optional[int] = 5,
        random_state: Optional[int] = 0,
        n_mc_samples: int = 100,
    ):
        if len(fidelity_levels) < 2:
            raise ValueError(
                "StackedVFMSurrogate requires at least 2 fidelity levels. "
                f"Got {len(fidelity_levels)}."
            )
        self.fidelity_levels = list(fidelity_levels)
        self.surrogate_cls = surrogate_cls
        self.surrogate_params = surrogate_params
        self.augment_with_std = augment_with_std
        self.output_dim_mismatch = output_dim_mismatch
        self.oof_cv = oof_cv
        self.random_state = random_state
        self.n_mc_samples = n_mc_samples

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_surrogate(self, level_idx: int):
        """Instantiate a fresh surrogate for ``level_idx``."""
        from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate

        if self.surrogate_cls is None:
            cls = RandomForestSurrogate
        elif isinstance(self.surrogate_cls, list):
            cls = self.surrogate_cls[level_idx]
        else:
            cls = self.surrogate_cls

        if self.surrogate_params is None:
            params: dict = {}
        elif isinstance(self.surrogate_params, list):
            params = dict(self.surrogate_params[level_idx])
        else:
            params = dict(self.surrogate_params)

        return cls(**params)

    def _augment(
        self,
        X: np.ndarray,
        level_idx: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Augment ``X`` with predictions from all levels < ``level_idx``.

        Chains augmentation sequentially: the augmented X for level i is
        used as input to surrogate i, whose predictions are then appended to
        produce the augmented X for level i+1.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features_in_)
            Raw (un-augmented) input.
        level_idx : int
            Target level index; predictions from levels 0..level_idx-1 are
            appended.
        rng : np.random.Generator or None
            If given, each lower-level prediction is replaced by one draw
            from ``N(pred, std)`` (Monte Carlo propagation, see ``predict``).

        Returns
        -------
        X_aug : np.ndarray
            Augmented feature matrix with shape
            (n_samples, augmented_n_features_per_level_[fidelity_levels[level_idx]]).
        """
        X_current = X  # shape (n_samples, n_features_in_)
        # Only set while fitting (see fit): out-of-fold predictors per level.
        oof = getattr(self, "_oof_predictors", None)

        for i in range(level_idx):
            level_name = self.fidelity_levels[i]
            surrogate_i = self.surrogates_[level_name]

            if oof is not None:
                pred_i, std_i = oof[level_name].predict(
                    surrogate_i, X, X_current, self.augment_with_std
                )
                aug_cols = np.hstack([pred_i, std_i]) if self.augment_with_std else pred_i
            elif self.augment_with_std or rng is not None:
                pred_i, std_i = surrogate_i.predict(X_current, return_std=True)
                pred_i, std_i = _as_2d(pred_i), _as_2d(std_i)
                if rng is not None:
                    pred_i = pred_i + std_i * rng.standard_normal(pred_i.shape)
                aug_cols = np.hstack([pred_i, std_i]) if self.augment_with_std else pred_i
            else:
                pred_i = surrogate_i.predict(X_current, return_std=False)
                pred_i = np.asarray(pred_i, dtype=np.float64)
                if pred_i.ndim == 1:
                    pred_i = pred_i.reshape(-1, 1)
                aug_cols = pred_i

            X_current = np.hstack([X_current, aug_cols])

        # Validate augmented feature count matches what was seen at fit time.
        # During fit, the entry for the current level doesn't exist yet — skip.
        target_level = self.fidelity_levels[level_idx]
        if target_level not in self.augmented_n_features_per_level_:
            return X_current

        expected = self.augmented_n_features_per_level_[target_level]
        actual = X_current.shape[1]

        if actual != expected:
            if self.output_dim_mismatch == "error":
                raise ValueError(
                    f"Augmented feature count mismatch for level '{target_level}': "
                    f"expected {expected} features, got {actual}. "
                    "Re-fit the model or set output_dim_mismatch='truncate'/'pad'."
                )
            elif self.output_dim_mismatch == "truncate":
                X_current = X_current[:, :expected]
            elif self.output_dim_mismatch == "pad":
                pad_width = expected - actual
                if pad_width > 0:
                    X_current = np.hstack(
                        [X_current, np.zeros((X_current.shape[0], pad_width))]
                    )
                else:
                    X_current = X_current[:, :expected]

        return X_current

    def _validate_data_dict(
        self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Check all required levels are present and arrays are consistent."""
        for level in self.fidelity_levels:
            if level not in data:
                raise ValueError(
                    f"Missing fidelity level '{level}' in data dict. "
                    f"Expected levels: {self.fidelity_levels}."
                )
            X, Y = data[level]
            X = np.asarray(X)
            Y = np.asarray(Y)
            if X.ndim != 2:
                raise ValueError(
                    f"X for level '{level}' must be 2D, got shape {X.shape}."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> "StackedVFMSurrogate":
        """Fit all level surrogates, augmenting features recursively.

        Parameters
        ----------
        data : dict[str, (X, Y)]
            Keys must match ``fidelity_levels`` (ordered lowest → highest).
            ``X`` has shape ``(n_samples, n_features)``; ``Y`` has shape
            ``(n_samples,)`` or ``(n_samples, n_outputs)``.
            Different levels may have different sample counts.

        Returns
        -------
        self
        """
        self._validate_data_dict(data)

        self.surrogates_: Dict[str, object] = {}
        self.n_outputs_per_level_: Dict[str, int] = {}
        self.augmented_n_features_per_level_: Dict[str, int] = {}
        self._oof_predictors: Dict[str, _OOFPredictor] = {}
        try:
            self._fit_levels(data)
        finally:
            # Fold models are only needed to build training features.
            del self._oof_predictors
        return self

    def _fit_levels(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
        last = len(self.fidelity_levels) - 1
        for k_idx, level in enumerate(self.fidelity_levels):
            X_k, Y_k = data[level]
            X_k = np.asarray(X_k, dtype=np.float64)
            Y_k = np.asarray(Y_k, dtype=np.float64)
            if Y_k.ndim == 1:
                Y_k = Y_k.reshape(-1, 1)

            if X_k.shape[0] < 5:
                warnings.warn(
                    f"Level '{level}' has only {X_k.shape[0]} samples. "
                    "Some surrogate types may fail or produce unreliable predictions.",
                    UserWarning,
                    stacklevel=2,
                )

            if k_idx == 0:
                self.n_features_in_ = X_k.shape[1]
                X_aug = X_k
            else:
                X_aug = self._augment(X_k, k_idx)

            surrogate_k = self._make_surrogate(k_idx)
            surrogate_k.fit(X_aug, Y_k)

            self.surrogates_[level] = surrogate_k
            self.n_outputs_per_level_[level] = Y_k.shape[1]
            self.augmented_n_features_per_level_[level] = X_aug.shape[1]
            if k_idx < last:
                self._oof_predictors[level] = _OOFPredictor(
                    lambda k=k_idx: self._make_surrogate(k),
                    X_k, X_aug, Y_k, self.oof_cv, self.random_state,
                )

    def predict(
        self,
        X: np.ndarray,
        level: Optional[str] = None,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict at a specified fidelity level.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Raw (un-augmented) input features.  Must match ``n_features_in_``.
        level : str or None
            Fidelity level to predict at.  Defaults to the highest level
            (last entry in ``fidelity_levels``).
        return_std : bool, default False
            If True, also return the predictive standard deviation. Lower-level
            uncertainty is propagated by Monte Carlo (``n_mc_samples`` draws):
            the std is the root mean square, over draws, of the top level's
            std and of its mean's deviation from ``y_pred``. ``y_pred`` itself
            is the plug-in prediction, the same as with ``return_std=False``.
            With ``n_mc_samples=0`` only the top level's own std is returned,
            which underestimates the total uncertainty.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]

        Raises
        ------
        ValueError
            If ``level`` is not in ``fidelity_levels``.
        AttributeError
            If the model has not been fitted.
        """
        if not hasattr(self, "surrogates_"):
            raise AttributeError(
                "StackedVFMSurrogate is not fitted. Call fit() first."
            )

        if level is None:
            level = self.fidelity_levels[-1]
        if level not in self.surrogates_:
            raise ValueError(
                f"Unknown fidelity level: '{level}'. "
                f"Available: {self.fidelity_levels}."
            )

        X = np.asarray(X, dtype=np.float64)
        level_idx = self.fidelity_levels.index(level)

        if level_idx == 0:
            X_aug = X
        else:
            X_aug = self._augment(X, level_idx)

        # Models pickled before n_mc_samples existed keep the old behaviour.
        n_mc = getattr(self, "n_mc_samples", 0)
        if not return_std or level_idx == 0 or not n_mc:
            return self.surrogates_[level].predict(X_aug, return_std=return_std)

        top = self.surrogates_[level]
        y_pred = top.predict(X_aug)
        y_pred_2d = _as_2d(y_pred)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        # Second moment about y_pred, E[sd^2 + (mu - y_pred)^2], accumulated
        # over chunks of draws to bound the rows predicted at once.
        sq_sum = np.zeros_like(y_pred_2d)
        chunk = max(1, _MC_MAX_ROWS // max(n, 1))
        for start in range(0, n_mc, chunk):
            k = min(chunk, n_mc - start)
            # One block of rows per draw; each block gets its own lower-level samples.
            X_draws = self._augment(np.tile(X, (k, 1)), level_idx, rng=rng)
            mu, sd = top.predict(X_draws, return_std=True)
            mu = _as_2d(mu).reshape(k, n, -1)
            sd = _as_2d(sd).reshape(k, n, -1)
            sq_sum += np.sum(sd ** 2 + (mu - y_pred_2d) ** 2, axis=0)
        y_std = np.sqrt(sq_sum / n_mc)
        return y_pred, y_std.reshape(np.shape(y_pred))

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        """Called by ConformalMixin; predicts at the highest fidelity level."""
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "fidelity_levels": self.fidelity_levels,
            "surrogate_cls": self.surrogate_cls,
            "surrogate_params": self.surrogate_params,
            "augment_with_std": self.augment_with_std,
            "output_dim_mismatch": self.output_dim_mismatch,
            "oof_cv": self.oof_cv,
            "random_state": self.random_state,
            "n_mc_samples": self.n_mc_samples,
        }

    def set_params(self, **params) -> "StackedVFMSurrogate":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AdditiveCorrectionVFM(ConformalMixin):
    """Two-level autoregressive correction (Kennedy-O'Hagan AR1) surrogate.

    The high-fidelity response is modelled as:

        f_hi(x) = rho * f_lo(x) + delta(x)

    ``rho`` (one per output) is the least-squares slope of ``Y_hi`` on
    ``f_lo(X_hi)``, and ``delta`` is a surrogate fitted to
    ``Y_hi - rho * f_lo(X_hi)``.  Both ``f_lo`` and ``delta`` are arbitrary
    surrogate models, so the correction can be nonlinear.  With a fixed
    ``rho=1`` the low-fidelity model is assumed to be on the same scale as the
    high-fidelity one, and any scale difference has to be learned by ``delta``
    from the (few) high-fidelity points.

    Uncertainty is combined in quadrature (assuming independence):

        sigma_hi(x) = sqrt(rho^2 * sigma_lo(x)^2 + sigma_delta(x)^2)

    Note: The independence assumption is approximate because ``delta`` is
    computed using ``f_lo``.  In practice the combined std slightly
    overestimates true uncertainty — a conservative, safe approximation.

    This class assumes ``Y_lo`` and ``Y_hi`` share the same output
    dimensionality.  For heterogeneous output spaces or more than two fidelity
    levels, use ``StackedVFMSurrogate``.

    Parameters
    ----------
    lo_surrogate_cls : type | None
        Surrogate class for the low-fidelity model.  Defaults to
        ``RandomForestSurrogate`` (fast, handles multi-output natively).
    hi_surrogate_cls : type | None
        Surrogate class for the correction (delta) model.  Defaults to
        ``GaussianProcessSurrogate`` — corrections are typically small and
        smooth, making a GP a natural prior.

        Warning: GP scales as O(n^3).  For hi-fi datasets with >500 samples,
        prefer a sparse GP (RFFGPSurrogate, NystroemGPSurrogate) or RF.
    lo_surrogate_params : dict | None
        Constructor kwargs for the low-fidelity surrogate.
    hi_surrogate_params : dict | None
        Constructor kwargs for the correction surrogate.
    rho : "fit" or float, default "fit"
        Scale on the low-fidelity prediction. ``"fit"`` estimates it per output
        by least squares (with an intercept, which ``delta`` absorbs); when the
        low-fidelity prediction is constant on the high-fidelity inputs it
        falls back to 1.  A number fixes it (``rho=1.0`` is the plain additive
        correction).  A fitted ``rho`` also multiplies the low-fidelity
        model's own error, so with a biased or strongly smoothed ``f_lo``
        (e.g. a random forest on few points) ``rho=1.0`` can predict better;
        compare both on held-out data.
    oof_cv : int or None, default 5
        Folds for out-of-fold ``f_lo`` predictions at high-fidelity inputs
        that are also low-fidelity training points (nested designs).  The
        full model is in-sample there, so ``rho`` and ``delta`` would be
        fitted against predictions more accurate than at new points.  ``None``
        or ``0`` uses the full model everywhere.
    random_state : int or None, default 0
        Seed for the out-of-fold split.

    Attributes
    ----------
    surrogate_lo_ : surrogate
        Fitted low-fidelity surrogate.
    surrogate_delta_ : surrogate
        Fitted correction surrogate.
    rho_ : np.ndarray, shape (n_outputs,)
        Scale applied to the low-fidelity prediction.
    n_outputs_ : int
        Number of outputs (must match between lo and hi data).

    Examples
    --------
    >>> import numpy as np
    >>> from multioutreg.surrogates import AdditiveCorrectionVFM
    >>> rng = np.random.default_rng(42)
    >>> X_lo = rng.standard_normal((100, 2))
    >>> Y_lo = np.sin(X_lo).sum(axis=1, keepdims=True)
    >>> X_hi = rng.standard_normal((25, 2))
    >>> Y_hi = np.sin(X_hi).sum(axis=1, keepdims=True) + 0.05 * X_hi[:, :1]
    >>> model = AdditiveCorrectionVFM()
    >>> model.fit({"lo": (X_lo, Y_lo), "hi": (X_hi, Y_hi)})
    AdditiveCorrectionVFM(...)
    >>> y_pred, y_std = model.predict(X_hi, return_std=True)
    >>> y_pred.shape
    (25, 1)
    """

    _multi_output = True

    def __init__(
        self,
        lo_surrogate_cls=None,
        hi_surrogate_cls=None,
        lo_surrogate_params: Optional[dict] = None,
        hi_surrogate_params: Optional[dict] = None,
        rho: Union[Literal["fit"], float] = "fit",
        oof_cv: Optional[int] = 5,
        random_state: Optional[int] = 0,
    ):
        self.lo_surrogate_cls = lo_surrogate_cls
        self.hi_surrogate_cls = hi_surrogate_cls
        self.lo_surrogate_params = lo_surrogate_params
        self.hi_surrogate_params = hi_surrogate_params
        self.rho = rho
        self.oof_cv = oof_cv
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_lo_surrogate(self):
        from multioutreg.surrogates.rf_sklearn import RandomForestSurrogate

        cls = self.lo_surrogate_cls if self.lo_surrogate_cls is not None else RandomForestSurrogate
        params = dict(self.lo_surrogate_params) if self.lo_surrogate_params else {}
        return cls(**params)

    def _make_hi_surrogate(self):
        from multioutreg.surrogates.gp_sklearn import GaussianProcessSurrogate

        cls = self.hi_surrogate_cls if self.hi_surrogate_cls is not None else GaussianProcessSurrogate
        params = dict(self.hi_surrogate_params) if self.hi_surrogate_params else {}
        return cls(**params)

    def _fit_rho(self, y_lo: np.ndarray, Y_hi: np.ndarray) -> np.ndarray:
        """Per-output scale on ``f_lo``: fixed, or the OLS slope of Y_hi on f_lo."""
        n_out = Y_hi.shape[1]
        if not (isinstance(self.rho, str) and self.rho == "fit"):
            return np.full(n_out, float(self.rho))
        rho = np.ones(n_out)
        yc = y_lo - y_lo.mean(axis=0)
        var = (yc ** 2).sum(axis=0)
        # f_lo (numerically) constant on X_hi: slope undefined, keep rho = 1.
        ok = var > 1e-12 * np.maximum((y_lo ** 2).sum(axis=0), 1e-300)
        cov = (yc * (Y_hi - Y_hi.mean(axis=0))).sum(axis=0)
        rho[ok] = cov[ok] / var[ok]
        return rho

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> "AdditiveCorrectionVFM":
        """Fit the low-fidelity surrogate and the additive correction model.

        Parameters
        ----------
        data : dict with keys "lo" and "hi"
            Each value is an (X, Y) tuple.  ``Y_lo`` and ``Y_hi`` must have
            the same number of columns (same output space).

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If required keys are missing or output dimensions differ.
        """
        for key in ("lo", "hi"):
            if key not in data:
                raise ValueError(
                    f"AdditiveCorrectionVFM requires 'lo' and 'hi' keys in data. "
                    f"Missing: '{key}'."
                )

        X_lo, Y_lo = data["lo"]
        X_hi, Y_hi = data["hi"]

        X_lo = np.asarray(X_lo, dtype=np.float64)
        X_hi = np.asarray(X_hi, dtype=np.float64)
        Y_lo = np.asarray(Y_lo, dtype=np.float64)
        Y_hi = np.asarray(Y_hi, dtype=np.float64)

        if Y_lo.ndim == 1:
            Y_lo = Y_lo.reshape(-1, 1)
        if Y_hi.ndim == 1:
            Y_hi = Y_hi.reshape(-1, 1)

        if Y_lo.shape[1] != Y_hi.shape[1]:
            raise ValueError(
                "AdditiveCorrectionVFM requires lo and hi to have the same "
                f"output dimension. Got lo={Y_lo.shape[1]}, hi={Y_hi.shape[1]}. "
                "Use StackedVFMSurrogate for heterogeneous output dimensions."
            )

        self.n_outputs_ = Y_lo.shape[1]
        self.n_features_in_ = X_lo.shape[1]

        self.surrogate_lo_ = self._make_lo_surrogate()
        self.surrogate_lo_.fit(X_lo, Y_lo)

        oof = _OOFPredictor(
            self._make_lo_surrogate, X_lo, X_lo, Y_lo, self.oof_cv, self.random_state
        )
        y_lo_at_hi, _ = oof.predict(self.surrogate_lo_, X_hi, X_hi, False)
        self.rho_ = self._fit_rho(y_lo_at_hi, Y_hi)
        delta = Y_hi - self.rho_ * y_lo_at_hi

        self.surrogate_delta_ = self._make_hi_surrogate()
        self.surrogate_delta_.fit(X_hi, delta)

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict high-fidelity output as rho * f_lo(x) + delta(x).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        return_std : bool, default False
            If True, return combined predictive std in quadrature:
            ``sigma_hi = sqrt(rho^2 * sigma_lo^2 + sigma_delta^2)``.

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples, n_outputs)
        y_std : np.ndarray, shape (n_samples, n_outputs)  [only if return_std]

        Raises
        ------
        AttributeError
            If the model has not been fitted.
        """
        if not hasattr(self, "surrogate_lo_"):
            raise AttributeError(
                "AdditiveCorrectionVFM is not fitted. Call fit() first."
            )

        X = np.asarray(X, dtype=np.float64)

        if return_std:
            y_lo, s_lo = self.surrogate_lo_.predict(X, return_std=True)
            delta_hat, s_delta = self.surrogate_delta_.predict(X, return_std=True)

            y_lo = np.asarray(y_lo, dtype=np.float64)
            delta_hat = np.asarray(delta_hat, dtype=np.float64)
            s_lo = np.asarray(s_lo, dtype=np.float64)
            s_delta = np.asarray(s_delta, dtype=np.float64)

            if y_lo.ndim == 1:
                y_lo = y_lo.reshape(-1, 1)
            if delta_hat.ndim == 1:
                delta_hat = delta_hat.reshape(-1, 1)
            if s_lo.ndim == 1:
                s_lo = s_lo.reshape(-1, 1)
            if s_delta.ndim == 1:
                s_delta = s_delta.reshape(-1, 1)

            y_pred = self.rho_ * y_lo + delta_hat
            y_std = np.sqrt((self.rho_ * s_lo) ** 2 + s_delta ** 2)
            return y_pred, y_std

        y_lo = np.asarray(self.surrogate_lo_.predict(X), dtype=np.float64)
        delta_hat = np.asarray(self.surrogate_delta_.predict(X), dtype=np.float64)

        if y_lo.ndim == 1:
            y_lo = y_lo.reshape(-1, 1)
        if delta_hat.ndim == 1:
            delta_hat = delta_hat.reshape(-1, 1)

        return self.rho_ * y_lo + delta_hat

    def predict_components(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rho * f_lo(x), delta(x)) separately for diagnostics.

        Useful for inspecting how much correction is applied and where.  The
        two parts sum to ``predict(X)``; ``rho_`` holds the scale.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_lo : np.ndarray, shape (n_samples, n_outputs)
            Scaled low-fidelity term ``rho * f_lo(x)``.
        delta : np.ndarray, shape (n_samples, n_outputs)
        """
        if not hasattr(self, "surrogate_lo_"):
            raise AttributeError(
                "AdditiveCorrectionVFM is not fitted. Call fit() first."
            )

        X = np.asarray(X, dtype=np.float64)

        y_lo = np.asarray(self.surrogate_lo_.predict(X), dtype=np.float64)
        delta = np.asarray(self.surrogate_delta_.predict(X), dtype=np.float64)

        if y_lo.ndim == 1:
            y_lo = y_lo.reshape(-1, 1)
        if delta.ndim == 1:
            delta = delta.reshape(-1, 1)

        return self.rho_ * y_lo, delta

    def _conformal_point_predict(self, X: np.ndarray) -> np.ndarray:
        """Called by ConformalMixin for conformal calibration."""
        preds = self.predict(X)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def get_params(self, deep: bool = True) -> dict:
        return {
            "lo_surrogate_cls": self.lo_surrogate_cls,
            "hi_surrogate_cls": self.hi_surrogate_cls,
            "lo_surrogate_params": self.lo_surrogate_params,
            "hi_surrogate_params": self.hi_surrogate_params,
            "rho": self.rho,
            "oof_cv": self.oof_cv,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> "AdditiveCorrectionVFM":
        for key, value in params.items():
            setattr(self, key, value)
        return self
