# PWSQC-py

A python implementation of the quality control for crowdsourced personal weather
station (PWS) rainfall observations described in

de Vos, L. W., Leijnse, H., Overeem, A., & Uijlenhoet, R. (2019). Quality control for crowdsourced personal weather stations to enable operational rainfall monitoring. _Geophysical Research Letters_, 46, 8820-8829. [https://doi.org/10.1029/2019GL083731](https://doi.org/10.1029/2019GL083731)

It reproduces the reference implementation in R
([PWSQC](https://github.com/LottedeVos/PWSQC)) on long format DataFrames.

## Usage

The library works on a long format DataFrame with one row per station and time
interval, where the timestamp indicates the **end** of the interval and the
rainfall is the amount that fell **since the previous interval** in mm.

| intern_id | date                      | precip |
| --------- | ------------------------- | ------ |
| 105504    | 2024-10-28 22:05:00+00:00 | 0.0    |
| 105504    | 2024-10-28 22:10:00+00:00 | 0.101  |

```python
import pandas as pd
import pwsqc

# possibility to modify default config here while creating an Instance
config = pwsqc.Config()

df = pwsqc.prepare_timeseries(pd.read_parquet('station_data.parquet'))
meta = pd.read_parquet('station_metadata.parquet')
neighbors = pwsqc.find_station_neighbors(meta, d=config.d)

df = pwsqc.faulty_zero_filter(df, neighbors, n_stat=config.n_stat, n_int=config.n_int)
df = pwsqc.high_influx_filter(df, neighbors, n_stat=config.n_stat)
df = pwsqc.station_outlier_filter(df, neighbors, n_stat=config.n_stat, dbc=config.dbc)
df = pwsqc.bias_correction(df, dbc=config.dbc, beta=config.beta)
df = pwsqc.apply_flags(df, strict=True)
```

Every filter adds one column to the DataFrame and leaves the input untouched:

| column      | added by                 | meaning                                                                                                                            |
| ----------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `FZflag`    | `faulty_zero_filter`     | faulty zeroes [info](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083731#grl59347-sec-0014)                           |
| `HIflag`    | `high_influx_filter`     | high influx [info](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083731#grl59347-sec-0015)                             |
| `SOflag`    | `station_outlier_filter` | station outlier [info](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083731#grl59347-sec-0016)                         |
| `bias`      | `station_outlier_filter` | median relative bias with the neighbors [info](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083731#grl59347-sec-0016) |
| `BCF`       | `bias_correction`        | bias correction factor of that interval [info](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL083731#grl59347-sec-0016) |
| `precip_qc` | `apply_flags`            | corrected rainfall, NaN where flagged                                                                                              |

Each flag is `0` (no error), `1` (error) or `-1` (not enough information to
determine the flag). `apply_flags(strict=False)` only discards the intervals
flagged with `1` ("filtered flex"), `apply_flags(strict=True)` discards the
intervals flagged with `-1` as well ("filtered strict").

The filters have to be applied in that order, the station outlier filter needs
the flags of the two preceding filters and the bias correction needs the output
of the station outlier filter.

## Parameters

All parameters are named as in the paper and default to the values of Table 1.
They are also documented and collected in `pwsqc.Config`.

| parameter | default | description                                                              |
| --------- | ------- | ------------------------------------------------------------------------ |
| `d`       | 10000   | range in m within which stations are considered neighbors                |
| `n_stat`  | 5       | minimum number of neighbors with an observation                          |
| `n_int`   | 6       | number of intervals a station has to report zero rainfall while it rains |
| `phi_a`   | 0.4     | rainfall threshold of the median of the neighbors in mm                  |
| `phi_b`   | 10      | rainfall threshold of the station itself in mm                           |
| `m_int`   | 4032    | number of intervals of the comparison window                             |
| `m_rain`  | 100     | number of nonzero rainfall intervals the window must contain             |
| `m_match` | 200     | minimum number of overlapping intervals of a neighbor                    |
| `gamma`   | 0.15    | threshold of the median correlation with the neighbors                   |
| `beta`    | 0.2     | relative threshold a change of the correction factor has to exceed       |
| `dbc`     | 1.24    | default bias correction factor of the network                            |
