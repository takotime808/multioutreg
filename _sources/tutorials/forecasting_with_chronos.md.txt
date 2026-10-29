# Time-Series Forecasting with Chronos

:::{admonition} About this tutorial
:class: note tutorial-meta

**Covers:** zero-shot probabilistic forecasts from the Chronos / Chronos-Bolt
foundation models, from the GUI, the CLI and Python. **Needs:** the `ts` extra.
**Runs in:** seconds per series on a CPU, after a one-time model download.
:::

Chronos models are pre-trained on a large collection of time series, so they forecast a
new series **without training** ("zero-shot"). They return quantiles rather than a single
line, which gives a forecast band you can plan against. Use them when you need a quick,
calibrated baseline; when you want models fitted to your own data and ranked, use
`multioutreg ts-pipeline` instead (see {doc}`../quickstart`).

## Install

```bash
pip install ".[ts]"
```

The extra adds `chronos-forecasting` and `torch`. The first forecast downloads the model
weights from Hugging Face; later runs use the local cache. A GPU (`cuda`) or Apple
silicon (`mps`) is used automatically when available.

## From the GUI

```bash
streamlit run multioutreg/gui/pages/05_Time_Series_Forecasting.py
```

Upload a CSV, pick the value column(s) and the horizon, and download the forecast.

## From the command line

```bash
multioutreg ts-forecast data.csv \
    --time-col Date \
    --value-cols revenue,orders \
    --horizon 24 \
    --model amazon/chronos-bolt-base \
    --quantiles 0.1,0.5,0.9 \
    --out forecast.csv
```

| Option | Meaning |
|---|---|
| `--value-cols` | Columns to forecast, each as its own series. Defaults to the first numeric column |
| `--time-col` | Timestamp column; used only to keep it out of the default value column (rows must already be in time order and evenly spaced) |
| `--horizon` | Number of future steps |
| `--model` | Any Chronos or Chronos-Bolt model name on Hugging Face, e.g. `amazon/chronos-bolt-small` for speed |
| `--quantiles` | Quantile levels to write |

The output is a tidy CSV with one row per series, step and quantile:

| Column | Meaning |
|---|---|
| `id` | The value column the row belongs to |
| `step` | Steps ahead, from 1 to `--horizon` |
| `quantile` | The quantile level |
| `value` | The forecast at that quantile |

The command also prints a one-line JSON summary of the file written, the series and the
quantiles. The 0.1 and 0.9 quantiles bound an 80% forecast interval; the 0.5 quantile is
the median forecast.

:::{note}
Chronos-**Bolt** models produce the quantiles 0.1, 0.2, …, 0.9 only. Other requested
levels (such as 0.05 or 0.95) are dropped, and the summary lists the levels actually
written. The original Chronos models sample trajectories and support any level.
:::

## From Python

```python
import pandas as pd
from multioutreg.time_series.chronos_adapter import ChronosForecaster

df = pd.read_csv("data.csv")
fc = ChronosForecaster(model_name="amazon/chronos-bolt-base")
fc.fit({"revenue": df["revenue"].dropna().to_numpy()})
res = fc.predict(prediction_length=24, quantiles=(0.1, 0.5, 0.9))

res.quantiles.shape   # (n_series, n_quantiles, horizon)
res.median()          # the 0.5 quantile for each series
```

## Things to know

- **Evenly spaced data.** The model sees a sequence of values, not timestamps. Aggregate
  irregular data (for example orders to daily totals) before forecasting; `ts-pipeline`
  does this with `--freq` and `--agg`.
- **Check it on your data.** Hold out the last `horizon` points, forecast them, and check
  how often they fall inside the 0.1–0.9 band (about 80% of the time is right).
