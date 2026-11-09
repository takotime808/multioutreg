# Copyright (c) 2026 takotime808

"""Joint (simultaneous) conformal prediction regions for multi-output regression.

``SplitConformalPredictor`` and ``CVPlusConformalPredictor`` calibrate each
output on its own, so each interval covers its output with probability
``1 - alpha`` but all of them together cover the whole output vector less
often (as low as ``1 - m * alpha`` for ``m`` outputs). The predictor here
calibrates one score per row instead, so the region it returns covers the
whole vector with probability ``1 - alpha``.

Scores (``score=``):

- ``"max"``: max-norm of the residuals, each divided by a per-output scale.
  The region is a hyperrectangle.
- ``"mahalanobis"``: Mahalanobis norm under the residual covariance
  (Johnstone & Cox 2021). The region is an ellipsoid that follows the
  correlation between outputs, so it is smaller than a rectangle when the
  residuals are correlated.
- ``"bonferroni"``: per-output intervals at ``alpha / m``. Valid without any
  shape estimate, but conservative; kept as the baseline the other two
  should beat.

The scales and covariance (the region's *shape*) are estimated on a share of
the calibration rows (``shape_size``) that the quantile never sees, so the
coverage guarantee holds whatever shape is estimated. ``shape_size=0`` uses
the training residuals instead, which keeps every calibration row for the
quantile but gives a poor shape for models that interpolate their training
data (noise-free GPs, 1-NN), whose training residuals are near zero.
"""

import math
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from multioutreg.conformal.base import BaseConformalPredictor
from multioutreg.conformal.cv_plus import _fresh_copy

_SCORES = ("max", "mahalanobis", "bonferroni")


class JointConformalPredictor(BaseConformalPredictor):
    """Split conformal regions with simultaneous coverage across outputs.

    Coverage guarantee: P(the whole output vector lies in the region)
    >= 1 - alpha for exchangeable data.

    Parameters
    ----------
    estimator : regressor
        Any regressor with fit/predict. Cloned (or deep-copied) internally.
    score : {"max", "mahalanobis", "bonferroni"}
        Nonconformity score, and so the shape of the region (see module
        docstring). Default ``"max"``.
    calibration_size : float
        Fraction of the data held out from model fitting (default 0.2).
    shape_size : float
        Fraction of the held-out rows used to estimate the per-output scales
        or the residual covariance; the rest set the conformal quantile.
        ``0`` estimates the shape from training residuals. Ignored for
        ``"bonferroni"``. Default 0.5.
    adaptive : bool
        If True, residuals are divided by the model's ``return_std`` before
        scoring, so the region grows and shrinks with the model's own
        uncertainty.
    ridge : float
        Added to the covariance diagonal, relative to its mean variance, so
        a singular covariance (outputs that are exact functions of each
        other) can still be inverted. Default 1e-6.
    random_state : int, optional
        Seed for the calibration and shape splits.
    """

    def __init__(
        self,
        estimator,
        score: str = "max",
        calibration_size: float = 0.2,
        shape_size: float = 0.5,
        adaptive: bool = False,
        ridge: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        super().__init__(estimator, random_state)
        self.score = score
        self.calibration_size = calibration_size
        self.shape_size = shape_size
        self.adaptive = adaptive
        self.ridge = ridge

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JointConformalPredictor":
        """Fit the model, estimate the region shape and store calibration scores.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,) or (n_samples, n_outputs)
        """
        if self.score not in _SCORES:
            raise ValueError(f"score must be one of {_SCORES}, got {self.score!r}")
        if not 0 <= self.shape_size < 1:
            raise ValueError(f"shape_size must be in [0, 1), got {self.shape_size}")
        y = self._ensure_2d(y)
        self._n_outputs = y.shape[1]

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_size,
            random_state=self.random_state,
        )

        self.model_ = _fresh_copy(self.estimator)
        if self._n_outputs == 1:
            self.model_.fit(X_train, y_train.ravel())
        else:
            self.model_.fit(X_train, y_train)

        z_cal = self._normalized_residuals(X_cal, y_cal)

        if self.score == "bonferroni":
            z_shape, z_q = None, z_cal
        elif self.shape_size == 0:
            z_shape, z_q = self._normalized_residuals(X_train, y_train), z_cal
        else:
            z_shape, z_q = train_test_split(
                z_cal, test_size=1 - self.shape_size, random_state=self.random_state,
            )

        if self.score == "max":
            scale = np.std(z_shape, axis=0)
            # A constant residual (an output the model fits exactly) gives no
            # scale; fall back to 1 so that output is still scored.
            self.scale_ = np.where(scale > 0, scale, 1.0)
        elif self.score == "mahalanobis":
            cov = np.atleast_2d(np.cov(z_shape, rowvar=False))
            jitter = self.ridge * max(float(np.mean(np.diag(cov))), 1e-12)
            self.cov_ = cov + jitter * np.eye(self._n_outputs)
            self.precision_ = np.linalg.inv(self.cov_)

        self.scores_ = self._score(z_q)
        return self

    def _normalized_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Signed residuals y - y_hat, divided by the model std if adaptive."""
        mean, std = self._predict_mean_std(X)
        z = y - mean
        if std is not None:
            z = z / std
        return z

    def _predict_mean_std(self, X: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.adaptive:
            mean, std = self.model_.predict(X, return_std=True)
            return self._ensure_2d(mean), np.clip(self._ensure_2d(std), 1e-12, None)
        return self._ensure_2d(self.model_.predict(X)), None

    def _score(self, z: np.ndarray) -> np.ndarray:
        """Nonconformity score per row; (n, n_outputs) for Bonferroni."""
        if self.score == "max":
            return np.max(np.abs(z) / self.scale_, axis=1)
        if self.score == "mahalanobis":
            return np.sqrt(np.einsum("ij,jk,ik->i", z, self.precision_, z))
        return np.abs(z)

    def _quantile(self, alpha: float) -> np.ndarray:
        """Conformal quantile: a scalar, or one per output for Bonferroni."""
        self._check_alpha(alpha)
        if self.score == "bonferroni":
            return np.array([
                self._conformal_quantile(self.scores_[:, j], alpha / self._n_outputs)
                for j in range(self._n_outputs)
            ])
        return np.asarray(self._conformal_quantile(self.scores_, alpha))

    def _half_widths(self, alpha: float) -> np.ndarray:
        """Half-widths of the bounding box in normalized units, shape (n_outputs,)."""
        q = self._quantile(alpha)
        if self.score == "max":
            return q * self.scale_
        if self.score == "mahalanobis":
            # The ellipsoid's extent along output j is q * sqrt(cov_jj).
            return q * np.sqrt(np.diag(self.cov_))
        return q

    def predict_interval(
        self,
        X: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simultaneous intervals: all outputs covered together with prob >= 1-alpha.

        For ``"max"`` and ``"bonferroni"`` these bounds are the region itself.
        For ``"mahalanobis"`` they are the ellipsoid's bounding box, which is
        valid but larger than the ellipsoid; use :meth:`contains` and
        :meth:`log_volume` to work with the ellipsoid directly.

        Returns
        -------
        y_lower, y_upper : np.ndarray
            Shape (n_samples,) for single output or (n_samples, n_outputs).
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")
        mean, std = self._predict_mean_std(X)
        half = self._half_widths(alpha)[np.newaxis, :]
        if std is not None:
            half = half * std
        y_lower, y_upper = mean - half, mean + half
        if self._n_outputs == 1:
            return y_lower.ravel(), y_upper.ravel()
        return y_lower, y_upper

    def contains(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Whether each row of ``y`` lies inside the joint region at ``X``.

        Returns
        -------
        np.ndarray of bool, shape (n_samples,)
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")
        z = self._normalized_residuals(X, self._ensure_2d(y))
        q = self._quantile(alpha)
        scores = self._score(z)
        if self.score == "bonferroni":
            return np.all(scores <= q[np.newaxis, :], axis=1)
        return scores <= q

    def log_volume(self, X: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        """Natural log of the region's volume at each row of ``X``.

        Compares region sizes across scores: at the same coverage, a smaller
        volume is a tighter region. ``inf`` when the quantile is unbounded.

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if not hasattr(self, "model_"):
            raise AttributeError("Predictor not fitted. Call fit() first.")
        _, std = self._predict_mean_std(X)
        n = X.shape[0]
        m = self._n_outputs
        log_std = np.sum(np.log(std), axis=1) if std is not None else np.zeros(n)
        if self.score == "mahalanobis":
            q = float(self._quantile(alpha))
            if not np.isfinite(q):
                return np.full(n, np.inf)
            # Volume of {z : z' P z <= q^2} = V_m * q^m * sqrt(det cov), with
            # V_m the volume of the unit m-ball.
            log_unit_ball = 0.5 * m * math.log(math.pi) - math.lgamma(0.5 * m + 1)
            _, logdet = np.linalg.slogdet(self.cov_)
            base = log_unit_ball + m * math.log(q) + 0.5 * logdet if q > 0 else -np.inf
            return base + log_std
        half = self._half_widths(alpha)
        if not np.all(np.isfinite(half)):
            return np.full(n, np.inf)
        with np.errstate(divide="ignore"):
            base = float(np.sum(np.log(2 * half)))
        return base + log_std
