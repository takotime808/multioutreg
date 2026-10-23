# Time-Series Forecasting with Chronos

This tutorial shows how to obtain **zero-shot probabilistic forecasts** using Chronos / Chronos-Bolt.

## Installation 
```bash
pip install .[ts]
```

## Usage

### UI
```bash
streamlit run multioutreg/gui/pages/05_Time_Series_Forecasting.py
```

### CLI

```bash
# Example 1
multioutreg ts-forecast data.csv \
    --value-cols y \
    --horizon 24 \
    --model amazon/chronos-bolt-base

# Example 2
multioutreg ts-forecast path/to/series.csv \
    --value-cols y \
    --horizon 24 \
    --model amazon/chronos-bolt-base \
    --quantiles 0.1,0.5,0.9 \
    --out forecast.csv
```

