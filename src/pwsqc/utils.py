import os
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple
from typing import TypeVar

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from numpy.typing import NDArray

_ScalarT = TypeVar('_ScalarT', bound=np.generic)


class Config(NamedTuple):
    """
    :param d: All stations within a range (d) around a given station are selected to
        compute the median rainfall over the surrounding area.
    :param n_stat: If fewer than ``n_stat`` neighboring stations with rainfall
        measurements are available, the median cannot be calculated and the FZ flag is
        set to -1
    :param n_int: The FZ flag is set to 1 if this median rainfall is larger than zero
        for at least ``n_int`` time intervals while the station itself reports zero
        rainfall. The FZ flag remains 1 until the station reports nonzero rainfall.
    :param phi_a: If the median does not exceed a threshold value (phi_a), the HI flag
        is set to 1 for any rainfall value from the station itself above threshold
        ``phi_b``. When the surrounding stations report moderate to heavy rainfall,
        the threshold becomes variable: for a median of ``phi_a`` or higher, the
        stations' HI flag is set to 1 when its measurements exceed median times
        ``phi_b``/``phi_a``. HI flag is set to -1 if fewer than ``n_stat`` neighboring
        stations report observations.
    :param phi_b: see ``phi_a``.
    :param m_int: To determine whether a station yields nonsensical measurements for
        that location, it is compared with time series of neighboring stations within a
        range (d). A previous period of mint intervals, or any longer interval where
        the station has at least ``m_rain`` intervals of nonzero rainfall measurements,
        is evaluated. There needs to be at least ``n_stat`` stations with at least
        ``m_match`` intervals overlapping with the evaluated station to compute the
        SO flag.
    :param m_rain: see ``m_int``.
    :param m_match: see ``m_int``.
    :param gamma: The r (equation (1)) and bias (equation (2)) with all neighboring
        stations are calculated. If the median of the r values falls short of threshold
        ``gammma``, the SO flag is set to 1.
    :param beta: If this threshold is exceeded, ``BCFnew`` is computed from the median
        of the bias values with the neighboring stations.
        If ``|log(BCFnew/BCFprev)| > log(1+β)``, this is deemed a systematic change for
        that station and BCFprev is replaced with the new value. This is hence a way to
        dynamically update BCF for individual stations.
    :param dbc: The default bias correction to address the fact that the Netatmo
        rain gauges have a general tendency to underestimate rainfall. DBC is a
        single-value one-off proxy of the correction needed for the overall PWS network
        bias and can be determined a priori by comparing network measurements over a
        period with typical rainfall for the local climate.
    """
    d: float = 10_000
    n_stat: int = 5
    n_int: int = 6
    phi_a: float = 0.4
    phi_b: float = 10
    m_int: int = 4032
    m_rain: int = 100
    m_match: int = 200
    gamma: float = 0.15
    beta: float = 0.2
    dbc: float = 1.24


def prepare_timeseries(
        df_raw: pd.DataFrame,
        freq: str = '5min',
        id_col: str = 'intern_id',
        date_col: str = 'date',
        precip_col: str = 'precip',
) -> pd.DataFrame:
    """Prepare a time series DataFrame to be ready for PWSQC processing.

    This includes:
    - Ensuring the DataFrame has a DatetimeIndex that is timezone aware.
    - Resampling the data to a uniform frequency (default: 5 minutes).
    - sorting the DataFrame by time and station ID.

    :param df_raw: Input DataFrame with a DatetimeIndex and precipitation data.
    :param freq: Frequency string for resampling (default is '5min').
    :return: Resampled DataFrame with a uniform time index.
    """
    # first sort by time and station id
    df_raw = df_raw.sort_values(by=[date_col, id_col]).copy()
    df_raw = df_raw.drop_duplicates(subset=[date_col, id_col, precip_col])
    # check if we have duplicated date and id values but differing precipitation values
    duplicates = df_raw.duplicated(subset=[date_col, id_col], keep=False)
    if duplicates.any():
        dup_df = df_raw[duplicates]
        raise ValueError(
            f"Duplicate date and id values with differing precipitation "
            f"values found:\n{dup_df}",
        )
    # now round the date to the defined frequency by aggregating to the specified
    # frequency. min_count keeps intervals without any observation missing, an
    # interval a station did not report is not the same as a reported zero
    df = (
        df_raw.set_index(date_col)
        .groupby(id_col)[precip_col]
        .resample(freq)
        .sum(min_count=1)
        .reset_index()
    )
    # reindex to have a full time series for each station
    full_time = pd.date_range(
        start=df[date_col].min(),
        end=df[date_col].max(),
        freq=freq,
        tz='UTC',
    )
    df = df.set_index([id_col, date_col])
    # all intern_ids
    interns = df.index.get_level_values(id_col).unique()

    # build full multiindex
    full_index = pd.MultiIndex.from_product(
        [interns, full_time],
        names=[id_col, date_col],
    )
    # reindex (missing combinations become NaN)
    df = df.reindex(full_index)
    # sort by index to ensure proper order
    df = df.sort_index()
    return df.reset_index()


def _compute_row_distances(
        args: tuple[int, list[tuple[float, float]]],
) -> tuple[int, NDArray[np.float64]]:
    """Compute geodesic distances from station `i` to all stations with index > i.

    Returns the row index and a full-length array of distances (np.nan for j <= i,
    filled in by the caller via symmetry).
    """
    i, coords = args
    n = len(coords)
    row = np.full(n, np.nan)
    lat_i, lon_i = coords[i]
    for j in range(i + 1, n):
        row[j] = geodesic((lat_i, lon_i), coords[j]).meters
    return i, row


def find_station_neighbors(
        station_metadata: pd.DataFrame,
        d: float = 10_000,
        max_neighbors: int | None = None,
        id_col: str = 'intern_id',
        lat_col: str = 'lat',
        lon_col: str = 'lon',
        n_jobs: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """For each station, find neighboring stations within distance `d`, computed
    in parallel.

    Stations at a distance of exactly zero are not considered neighbors. This
    excludes the station itself, but also -- as in the reference implementation --
    any other station that reports the exact same coordinates.

    :param station_metadata: DataFrame containing station metadata with latitude
        and longitude.
    :param d: Distance threshold in meters to consider stations as neighbors.
    :param max_neighbors: If given, only the ``max_neighbors`` nearest stations
        within ``d`` are kept. This bounds the runtime of the filters in very
        dense networks.
    :param id_col: Column name for station identifier.
    :param lat_col: Column name for latitude.
    :param lon_col: Column name for longitude.
    :param n_jobs: Number of worker processes to use. Defaults to os.cpu_count().
    :return: Dict mapping each station id to a tuple of neighbor ids, sorted
        ascending by distance.
    """
    required_cols = {id_col, lat_col, lon_col}
    missing_cols = required_cols - set(station_metadata.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if d < 0:
        raise ValueError('d must be non-negative')

    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError('max_neighbors must be at least 1')

    # Work on a copy and remove rows with missing coordinates.
    stations = station_metadata[[id_col, lat_col, lon_col]].copy()
    stations = stations.dropna(subset=[lat_col, lon_col])
    stations = stations.drop_duplicates(subset=[id_col]).reset_index(drop=True)

    n = len(stations)
    ids = stations[id_col].to_numpy()
    coords: list[tuple[float, float]] = list(zip(stations[lat_col], stations[lon_col]))

    if n < 2:
        return {station_id: () for station_id in ids}

    if n_jobs is None:
        n_jobs = os.cpu_count() or 1

    # Compute the upper triangle in parallel, then mirror it. Diagonal stays 0.
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    tasks = [(i, coords) for i in range(n - 1)]  # last row has nothing left to compute

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for i, row in executor.map(_compute_row_distances, tasks):
            dist_matrix[i, i + 1:] = row[i + 1:]

    dist_matrix += dist_matrix.T  # mirror upper triangle onto lower triangle

    result: dict[int, tuple[int, ...]] = {}

    for i in range(n):
        distances = dist_matrix[i]
        order = np.argsort(distances, kind='stable')
        within = order[(distances[order] > 0) & (distances[order] <= d)]
        if max_neighbors is not None:
            within = within[:max_neighbors]
        result[ids[i]] = tuple(ids[j] for j in within)

    return result


def _to_wide(
        data: pd.DataFrame,
        id_col: str,
        date_col: str,
        value_cols: tuple[str, ...],
) -> tuple[list[NDArray[np.float64]], pd.DatetimeIndex, NDArray[np.int64]]:
    """Reshape the long format data into ``(time x station)`` matrices.

    The filters operate on regular time series where consecutive rows are
    consecutive measurement intervals, so the time axis is validated to be
    strictly increasing with a constant step.

    :param data: Long format DataFrame with one row per station and time step.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param value_cols: Columns to reshape, one matrix is returned per column.
    :return: A list of ``(n_times, n_stations)`` matrices, the shared time index
        and the station ids in the order of the matrix columns.
    """
    missing_cols = ({id_col, date_col} | set(value_cols)) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if data.duplicated(subset=[date_col, id_col]).any():
        raise ValueError(
            f'duplicate ({date_col}, {id_col}) combinations are not allowed, '
            f'use prepare_timeseries to build a regular time series',
        )

    wide = data.pivot(index=date_col, columns=id_col, values=list(value_cols))
    wide = wide.sort_index()

    times = pd.DatetimeIndex(wide.index)
    station_ids = wide[value_cols[0]].columns.to_numpy()

    if len(times) > 1:
        steps = np.unique(np.diff(times.to_numpy()))
        if len(steps) != 1:
            raise ValueError(
                f'the time series is not regular, found {len(steps)} different '
                f'time steps, use prepare_timeseries to build a regular time series',
            )

    matrices = [
        wide[col].to_numpy(dtype=np.float64, na_value=np.nan) for col in value_cols
    ]
    return matrices, times, station_ids


def _to_long(
        data: pd.DataFrame,
        values: NDArray[_ScalarT],
        times: pd.DatetimeIndex,
        station_ids: NDArray[np.int64],
        id_col: str,
        date_col: str,
) -> NDArray[_ScalarT]:
    """Map a ``(time x station)`` matrix back onto the rows of ``data``."""
    rows = times.get_indexer(pd.DatetimeIndex(data[date_col]))
    cols = pd.Index(station_ids).get_indexer(pd.Index(data[id_col]))
    return values[rows, cols]


def _neighbor_index(
        neighbors: Mapping[int, Sequence[int]],
        station_ids: NDArray[np.int64],
) -> list[NDArray[np.intp]]:
    """Translate the neighbor ids into column indices of the wide matrices.

    Neighbors without observations in the data set are silently dropped, they
    cannot contribute to the median or to the correlations either way.
    """
    lookup = pd.Index(station_ids)
    index = []
    for station_id in station_ids:
        ids = np.asarray(neighbors.get(station_id, ()))
        positions = lookup.get_indexer(pd.Index(ids)) if len(ids) else np.array([])
        index.append(positions[positions >= 0].astype(np.intp))
    return index


def _neighbor_stats(
        values: NDArray[np.float64],
        neighbor_index: list[NDArray[np.intp]],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute the median and the number of reporting neighbors per interval.

    :param values: ``(n_times, n_stations)`` matrix of rainfall observations.
    :param neighbor_index: Column indices of the neighbors of each station.
    :return: The ``(n_times, n_stations)`` median of the neighboring stations and
        the number of neighboring stations reporting an observation.
    """
    med = np.full(values.shape, np.nan)
    cnt = np.zeros(values.shape, dtype=np.int64)
    for i, columns in enumerate(neighbor_index):
        if len(columns) == 0:
            continue
        subset = values[:, columns]
        cnt[:, i] = np.count_nonzero(~np.isnan(subset), axis=1)
        # rows without any observation would warn about an all-NaN slice
        any_obs = cnt[:, i] > 0
        med[any_obs, i] = np.nanmedian(subset[any_obs], axis=1)
    return med, cnt


def _rle(x: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:  # noqa: E501
    """Run length encoding of ``x`` with the semantics of R's ``rle()``.

    A NaN is never equal to anything, not even to another NaN, so NaN values
    always interrupt a run and always form a run of length one.

    :return: The start index, the end index (inclusive) and the value of each run.
    """
    n = x.size
    if n == 0:
        empty_i = np.array([], dtype=np.intp)
        return empty_i, empty_i, np.array([], dtype=np.float64)
    # NaN != NaN evaluates to True in numpy, exactly what R's rle() does with NA
    changed = x[1:] != x[:-1]
    ends = np.append(np.flatnonzero(changed), n - 1)
    starts = np.append(0, ends[:-1] + 1)
    return starts, ends, x[ends]
