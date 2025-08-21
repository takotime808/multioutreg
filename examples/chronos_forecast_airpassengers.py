# Copyright (c) 2025 takotime808

import pandas as pd
from multioutreg.time_series.chronos_adapter import ChronosForecaster

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

y = df["#Passengers"].to_numpy()
f = ChronosForecaster("amazon/chronos-bolt-base").fit(y)

res = f.predict(prediction_length=12, quantiles=(0.1, 0.5, 0.9))

print("median month 1..12:", res.median[0, :])
