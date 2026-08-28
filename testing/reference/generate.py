"""Generate the reference data set the python implementation is validated with.

The synthetic PWS network contains all the errors the quality control targets,
as well as the situations in which a flag cannot be determined. The expected
flags are produced by the R reference implementation, see ``ref_qc.R``:

    python testing/reference/generate.py
    Rscript testing/reference/ref_qc.R testing/reference \
        10000 5 6 0.4 10 100 20 30 0.15 1.24 0.2
    python testing/reference/generate.py --compress

The parameters are the ones of ``tests/test_reference.py``, the comparison
windows are shortened so that the SO filter and the bias correction produce
results on a data set that is small enough to be kept in the repository.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
N_STATIONS = 30
N_TIMES = 400
SEED = 20190815
OUTPUTS = ('FZ_flags', 'HI_flags', 'SO_flags', 'BCF')


def build() -> None:
    rng = np.random.default_rng(SEED)

    # a dense network with three stations too far away to have any neighbors
    lat = 52.0 + rng.uniform(0, 0.10, N_STATIONS)
    lon = 4.80 + rng.uniform(0, 0.15, N_STATIONS)
    lat[:3] += np.array([1.0, 2.0, 3.0])

    # a few rainfall events passing over the area
    event = np.zeros(N_TIMES)
    for _ in range(8):
        start = rng.integers(0, N_TIMES - 40)
        length = rng.integers(10, 40)
        event[start:start + length] += rng.gamma(2.0, 0.6, length)

    field = np.maximum(
        event[:, None] * rng.gamma(3.0, 1 / 3, (N_TIMES, N_STATIONS)) +
        rng.normal(0, 0.02, (N_TIMES, N_STATIONS)),
        0.0,
    )
    field[field < 0.05] = 0.0
    # every station has its own bias and reports multiples of its bucket volume
    bias = rng.uniform(0.6, 1.5, N_STATIONS)
    data = np.round(field * bias / 0.101) * 0.101

    # station outliers: rainfall unrelated to the surrounding area
    for i in (5, 6):
        outlier = rng.gamma(1.0, 0.5, N_TIMES) * (rng.random(N_TIMES) < 0.25)
        data[:, i] = np.round(outlier / 0.101) * 0.101
    # faulty zeroes: the gauge is blocked and actively reports zeroes
    for column, (first, last) in ((7, (100, 250)), (8, (150, 380)), (9, (0, 60))):
        data[first:last, column] = 0.0
    # high influx: someone poured a bucket of water through the gauge
    for column, t in ((10, 50), (10, 51), (11, 300), (12, 120)):
        data[t, column] = 25.0
    # data gaps of every shape
    data[rng.random((N_TIMES, N_STATIONS)) < 0.05] = np.nan
    data[200:260, 13] = np.nan       # a longer outage
    data[:, 14] = np.nan             # a station without any observation
    data[:N_TIMES // 2, 15] = np.nan  # a station that starts halfway
    data[:, 16] = 0.0                # a station that only reports zeroes

    times = pd.date_range('2024-05-01 00:05', periods=N_TIMES, freq='5min', tz='UTC')
    ids = np.arange(1000, 1000 + N_STATIONS)
    pd.DataFrame(data, index=times, columns=ids).to_csv(
        os.path.join(HERE, 'Ndataset.csv'),
        index_label='time_end',
    )

    # neighbor list with the haversine distance of the reference implementation
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    d_lat = lat_r[:, None] - lat_r[None, :]
    d_lon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(d_lat / 2) ** 2 +
        np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(d_lon / 2) ** 2
    )
    distance = 6378137.0 * 2 * np.arcsin(np.sqrt(a))
    rows = []
    for i in range(N_STATIONS):
        within = np.flatnonzero((distance[i] > 0) & (distance[i] <= 10000))
        within = within[np.argsort(distance[i][within])]
        rows.append({
            'station_id': ids[i],
            'neighbours': ','.join(str(ids[j]) for j in within),
        })
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'neighbourlist.csv'), index=False)


def compress() -> None:
    """Store the in- and output of the R reference implementation compressed."""
    for name in ('Ndataset', *OUTPUTS):
        path = os.path.join(HERE, f'{name}.csv')
        df = pd.read_csv(path, float_precision='round_trip')
        df.to_csv(f'{path}.gz', index=False)
        os.remove(path)


def main() -> int:
    if '--compress' in sys.argv:
        compress()
    else:
        build()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
