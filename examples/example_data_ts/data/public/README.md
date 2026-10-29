# Public time series for testing `ts-pipeline`

Real, openly available univariate series, cleaned to two columns (`date,<target>`).
Regenerate with `python examples/example_data_ts/fetch_public_datasets.py`.

| File | Rows | Span | `--target-col` | `--freq` | What it exercises |
|---|---:|---|---|---|---|
| `airpassengers_monthly.csv` | 144 | 1949–1960 | `passengers` | `MS` | Trend + multiplicative yearly seasonality; short series |
| `co2_mauna_loa_monthly.csv` | ~820 | 1958–present | `co2_ppm` | `MS` | Smooth trend + strong period-12 cycle; a few missing months dropped |
| `melbourne_min_temp_daily.csv` | 3650 | 1981–1990 | `min_temp_c` | `1D` | Long daily series, period-365 seasonality, 2 missing days (31 Dec of leap years) |
| `female_births_daily.csv` | 365 | 1959 | `births` | `1D` | Near-white-noise; the naive baseline should be hard to beat |
| `sunspots_monthly.csv` | 2820 | 1749–1983 | `sunspots` | `MS` | ~11-year (≈132-month) non-calendar cycle, non-negative and spiky |
| `bike_sharing_daily.csv` | 731 | 2011–2012 | `rentals` | `1D` | Weekly + yearly pattern with growth |
| `bike_sharing_hourly.csv` | 17379 | 2011–2012 | `rentals` | `1h` | High frequency, daily/weekly seasonality, 165 missing hours to fill |

Example:

```bash
mor ts-pipeline examples/example_data_ts/data/public/airpassengers_monthly.csv \
    --datetime-col date --target-col passengers --freq MS
```

## Sources and licences

- **AirPassengers**: Box, Jenkins & Reinsel, *Time Series Analysis* (1976), via R `datasets`; mirrored from
  [AileenNielsen/TimeSeriesAnalysisWithPython](https://github.com/AileenNielsen/TimeSeriesAnalysisWithPython).
- **Mauna Loa CO₂**: NOAA Global Monitoring Laboratory and Scripps Institution of Oceanography,
  <https://gml.noaa.gov/ccgg/trends/>. Free to use, with citation requested.
- **Melbourne daily minimum temperature**: Australian Bureau of Meteorology, via
  [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets).
- **Daily female births, California 1959**: Hipel & McLeod (1994), via jbrownlee/Datasets.
- **Monthly sunspots**: WDC-SILSO, Royal Observatory of Belgium (CC BY-NC 4.0), historical version via jbrownlee/Datasets.
- **Bike sharing (Capital Bikeshare)**: Fanaee-T & Gama (2013), UCI ML Repository
  <https://doi.org/10.24432/C5W894> (CC BY 4.0).
