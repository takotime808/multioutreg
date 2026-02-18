# Copyright (c) 2025 takotime808

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

class BaseSurrogate:
    """Base class for multi-output surrogates."""
    def __init__(self, base_estimator):
        self.model = MultiOutputRegressor(base_estimator)

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X, return_std=False):
        if return_std:
            try:
                return self.model.predict(X, return_std=True)
            except TypeError:
                preds = self.model.predict(X)
                std = np.zeros_like(preds)
                return preds, std
        return self.model.predict(X)

    def wrap_conformal(self, X_cal, y_cal, method="split", **kwargs):
        """Attach a conformal predictor calibrated on the given data.

        Parameters
        ----------
        X_cal : np.ndarray
            Calibration features (must NOT overlap with training data).
        y_cal : np.ndarray
            Calibration targets.
        method : str
            "split" for SplitConformalPredictor or "cv+" for CVPlusConformalPredictor.
        **kwargs
            Additional arguments passed to the conformal predictor constructor.

        Returns
        -------
        self
        """
        from multioutreg.conformal import SplitConformalPredictor, CVPlusConformalPredictor

        if method == "split":
            self._conformal = SplitConformalPredictor(self.model, **kwargs)
        elif method == "cv+":
            self._conformal = CVPlusConformalPredictor(self.model, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'split' or 'cv+'.")
        self._conformal.fit(X_cal, y_cal)
        return self

    def conformal_predict(self, X, alpha=0.1):
        """Return conformal prediction intervals.

        Parameters
        ----------
        X : np.ndarray
        alpha : float
            Miscoverage level. Intervals target 1-alpha coverage.

        Returns
        -------
        y_lower, y_upper : np.ndarray
        """
        if not hasattr(self, '_conformal'):
            raise AttributeError(
                "No conformal predictor attached. Call wrap_conformal() first."
            )
        return self._conformal.predict_interval(X, alpha)