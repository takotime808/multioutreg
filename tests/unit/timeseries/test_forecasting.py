import numpy as np
from sklearn.linear_model import LinearRegression

from multioutreg.timeseries import TimeSeriesForecaster


def test_forecast_linear_series():
    """TimeSeriesForecaster should predict the continuation of a linear trend."""
    series = np.arange(10, dtype=float)
    # Train on first 8 points, forecast next 2
    train_series = series[:-2]
    forecaster = TimeSeriesForecaster(LinearRegression(), lags=3, horizon=2)
    forecaster.fit(train_series)
    preds = forecaster.predict(train_series)
    expected = series[-2:]
    assert preds.shape == (2,)
    assert np.allclose(preds, expected, atol=1e-5)
