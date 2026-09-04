import os
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import NamedTuple
from typing import TypeVar

import numpy as np
from geopy.distance import geodesic
from numpy.typing import NDArray

from . import _frames
from ._frames import FrameT

_ScalarT = TypeVar('_ScalarT', bound=np.generic)
_T = TypeVar('_T')

# pandas style frequency aliases that polars does not know, they are translated
# so that the frequency strings that were used before keep working
_FREQ_ALIASES = {
    'min': 'm',
    'T': 'm',
    'H': 'h',
    'S': 's',
    'L': 'ms',
    'U': 'us',
    'N': 'ns',
    'D': 'd',
    'W': 'w',
    'M': 'mo',
    'Y': 'y',
}


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


def _workers(n_jobs: int | None, n_items: int) -> int:
    """The number of workers to spread ``n_items`` pieces of work over."""
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    return max(1, min(n_jobs, n_items))


def _run(
        function: Callable[[_T], Any],
        items: Sequence[_T],
        n_jobs: int | None,
) -> None:
    """Call ``function`` for every item, in threads if that is worth it.

    The per station work of the filters is numpy heavy and numpy releases the
    GIL, so threads actually run in parallel here. Every call writes to its own
    slice of the result, so the order they run in does not matter.
    """
    workers = _workers(n_jobs, len(items))
    if workers == 1:
        for item in items:
            function(item)
        return
    with ThreadPoolExecutor(workers) as executor:
        # consume the iterator so that an exception of a worker is raised
        for _ in executor.map(function, items):
            pass


def _row_nanmedian(
        values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Row wise median and count of the values that are not NaN.

    Equivalent to ``np.nanmedian(values, axis=1)`` paired with a count of the
    observations, but several times faster: ``np.nanmedian`` falls back to a
    Python level loop over the rows as soon as the input contains a NaN. Sorting
    moves the NaN values to the end of every row, so the median is the middle of
    the leading valid part.

    :param values: ``(n_rows, n_columns)`` matrix, may contain NaN.
    :return: The median of every row, NaN for a row without any value, and the
        number of values that are not NaN per row.
    """
    n_rows, n_columns = values.shape
    if n_columns == 0:
        return (
            np.full(n_rows, np.nan),
            np.zeros(n_rows, dtype=np.int64),
        )
    # numpy sorts NaN to the end of the row
    ordered = np.sort(values, axis=1)
    count = n_columns - np.count_nonzero(np.isnan(ordered), axis=1)
    rows = np.arange(n_rows)
    # for an even count both middle values are averaged, for an odd count the
    # two indices are the same and the middle value is taken as it is
    lower = ordered[rows, (count - 1) // 2]
    upper = ordered[rows, count // 2]
    median = np.where(count > 0, 0.5 * (lower + upper), np.nan)
    return median, count.astype(np.int64)


def _duplicate_runs(
        dates: NDArray[np.generic],
        ids: NDArray[np.generic],
        values: NDArray[np.generic],
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Find the repeated ``(date, id)`` rows of a frame sorted by date and id.

    Repeated combinations are neighbors once the rows are sorted, so one pass
    over them is enough. Hashing the whole frame twice -- once to drop the exact
    duplicates and once to find the conflicting ones -- is by far the most
    expensive part of preparing a long time series otherwise.

    :param dates: The (interval end) timestamps, ascending.
    :param ids: The station ids, ascending within a timestamp.
    :param values: The rainfall of the row.
    :return: A mask of the first row of every ``(date, id)`` combination and a
        mask of the rows that report a different value than the first row of
        their combination.
    """
    n = dates.size
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    repeated = (dates[1:] == dates[:-1]) & (ids[1:] == ids[:-1])
    first = np.concatenate((np.ones(1, dtype=bool), ~repeated))
    # the value of the first row of the combination every row belongs to
    starts = np.flatnonzero(first)
    lengths = np.diff(np.append(starts, n))
    reference = values[np.repeat(starts, lengths)]
    equal = values == reference
    if values.dtype.kind == 'f':
        # an exact duplicate of a missing value is a duplicate, not a conflict
        equal |= np.isnan(values) & np.isnan(reference)
    return first, ~equal


def _conflicting_rows(
        first: NDArray[np.bool_],
        conflicting: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Every row of a ``(date, id)`` combination that reports differing values."""
    starts = np.flatnonzero(first)
    lengths = np.diff(np.append(starts, first.size))
    run_of_row = np.repeat(np.arange(starts.size), lengths)
    bad = np.bincount(
        run_of_row[conflicting], minlength=starts.size,
    ).astype(bool)
    return bad[run_of_row]


def _station_constants(
        df_raw: Any,
        columns: list[str],
        id_col: str,
        freq: str,
) -> Any:
    """Reduce the columns that are constant within a station to one row each.

    :param df_raw: Input DataFrame in long format.
    :param columns: Columns to reduce.
    :param id_col: Column name for station identifier.
    :param freq: Frequency the time series is resampled to, only used for the
        error message.
    :raises ValueError: if one of the columns holds more than one value within a
        station, it cannot be carried through unchanged then.
    :return: One row per station with the columns and the station id.
    """
    if _frames.is_polars(df_raw):
        return _station_constants_polars(df_raw, columns, id_col, freq)

    grouped = df_raw.groupby(id_col, sort=False)
    varying = [
        col for col in columns if (grouped[col].nunique(dropna=True) > 1).any()
    ]
    if varying:
        raise ValueError(
            f'the columns {varying} are not constant within a station and cannot '
            f'be carried through unchanged, aggregate them to {freq!r} yourself '
            f'or exclude them using keep_cols',
        )
    # first() skips the missing values, a station without any value keeps NaN
    return grouped[columns].first().reset_index()


def _is_missing(column: str, dtype: Any) -> Any:
    """Expression that is true where ``column`` holds no value.

    A NaN of a floating point column counts as missing as well, it is what a
    column built from a numpy array carries where a value is absent.
    """
    import polars as pl

    expr = pl.col(column).is_null()
    if dtype.is_float():
        expr = expr | pl.col(column).is_nan()
    return expr


def _valid(column: str, dtype: Any) -> Any:
    """The values of ``column`` that are neither null nor NaN."""
    import polars as pl

    return pl.col(column).filter(~_is_missing(column, dtype))


def _station_constants_polars(
        df_raw: Any,
        columns: list[str],
        id_col: str,
        freq: str,
) -> Any:
    """The polars flavour of :func:`_station_constants`."""
    schema = df_raw.schema
    grouped = df_raw.group_by(id_col, maintain_order=True).agg([
        _valid(col, schema[col]).n_unique().alias(col) for col in columns
    ])
    varying = [col for col in columns if (grouped[col] > 1).any()]
    if varying:
        raise ValueError(
            f'the columns {varying} are not constant within a station and cannot '
            f'be carried through unchanged, aggregate them to {freq!r} yourself '
            f'or exclude them using keep_cols',
        )
    # the missing values are skipped, a station without any value keeps null
    return df_raw.group_by(id_col, maintain_order=True).agg([
        _valid(col, schema[col]).first().alias(col) for col in columns
    ])


def _interval(freq: str) -> str:
    """Translate a frequency string into a polars duration string.

    Polars spells the unit of a minute ``m``, the pandas aliases such as ``5min``
    are accepted as well so that existing code keeps working.

    :param freq: Frequency string, e.g. ``5min``, ``5m`` or ``1h``.
    """
    digits = len(freq) - len(freq.lstrip('0123456789'))
    count, unit = freq[:digits], freq[digits:]
    return f'{count}{_FREQ_ALIASES.get(unit, unit)}'


def prepare_timeseries(
        df_raw: FrameT,
        freq: str = '5min',
        id_col: str = 'intern_id',
        date_col: str = 'date',
        precip_col: str = 'precip',
        keep_cols: Sequence[str] | None = None,
) -> FrameT:
    """Prepare a time series DataFrame to be ready for PWSQC processing.

    This includes:
    - Ensuring the timestamp column is timezone aware.
    - Resampling the data to a uniform frequency (default: 5 minutes).
    - sorting the DataFrame by time and station ID.

    The rows of the result are not the rows of the input: observations within the
    same interval are summed up and the intervals a station did not report are
    added as missing values. An additional column can hence only be carried
    through unchanged if it is constant within a station, which is the case for
    station metadata such as a location or a city id.

    :param df_raw: Input DataFrame with a timestamp column and precipitation
        data, either a :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param freq: Frequency string for resampling (default is '5min'). Both the
        pandas aliases and the polars duration strings are understood.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param precip_col: Column name holding the rainfall of the interval in mm.
    :param keep_cols: Additional columns to carry through unchanged, they have to
        be constant within a station. Defaults to every additional column of
        ``df_raw``, pass an empty sequence to drop them all.
    :return: Resampled DataFrame with a uniform time index, of the same type as
        ``df_raw``.
    """
    reserved = (id_col, date_col, precip_col)
    if keep_cols is None:
        requested = set(_frames.columns(df_raw)) - set(reserved)
    else:
        requested = set(keep_cols) - set(reserved)
        missing_cols = requested - set(_frames.columns(df_raw))
        if missing_cols:
            raise ValueError(f"Missing required columns: {sorted(missing_cols)}")
    # keep the columns in the order they were given in
    kept = [col for col in _frames.columns(df_raw) if col in requested]

    if _frames.is_polars(df_raw):
        return _prepare_timeseries_polars(
            df_raw, freq, id_col, date_col, precip_col, kept,
        )
    return _prepare_timeseries_pandas(
        df_raw, freq, id_col, date_col, precip_col, kept,
    )


def _prepare_timeseries_pandas(
        df_raw: Any,
        freq: str,
        id_col: str,
        date_col: str,
        precip_col: str,
        kept: list[str],
) -> Any:
    """The pandas flavour of :func:`prepare_timeseries`."""
    import pandas as pd

    # first sort by time and station id
    df_raw = df_raw.sort_values(by=[date_col, id_col])
    # the additional columns are checked before the deduplication, dropping rows
    # could hide that a column is not constant
    constants = _station_constants(
        df_raw=df_raw,
        columns=kept,
        id_col=id_col,
        freq=freq,
    )
    first, conflicting = _duplicate_runs(
        dates=_frames.values(df_raw, date_col),
        ids=_frames.values(df_raw, id_col),
        values=_frames.values(df_raw, precip_col),
    )
    if conflicting.any():
        # only the offending combinations are reported, they are few and the
        # exact duplicates among them are dropped as they were before
        dup_df = df_raw[_conflicting_rows(first, conflicting)].drop_duplicates(
            subset=[date_col, id_col, precip_col],
        )
        raise ValueError(
            f"Duplicate date and id values with differing precipitation "
            f"values found:\n{dup_df}",
        )
    df_raw = df_raw[first]
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
    df = df.reset_index()
    if kept:
        # every station of the result has a row in the constants, so this only
        # broadcasts the values and never introduces a missing value
        df = df.merge(constants, on=id_col, how='left', validate='many_to_one')
    return df


def _prepare_timeseries_polars(
        df_raw: Any,
        freq: str,
        id_col: str,
        date_col: str,
        precip_col: str,
        kept: list[str],
) -> Any:
    """The polars flavour of :func:`prepare_timeseries`."""
    import polars as pl

    # first sort by time and station id
    df_raw = df_raw.sort(by=[date_col, id_col], maintain_order=True)
    # the additional columns are checked before the deduplication, dropping rows
    # could hide that a column is not constant
    constants = _station_constants(
        df_raw=df_raw,
        columns=kept,
        id_col=id_col,
        freq=freq,
    )
    first, conflicting = _duplicate_runs(
        dates=_frames.values(df_raw, date_col),
        ids=_frames.values(df_raw, id_col),
        values=_frames.values(df_raw, precip_col),
    )
    if conflicting.any():
        # only the offending combinations are reported, they are few and the
        # exact duplicates among them are dropped as they were before
        dup_df = df_raw.filter(_conflicting_rows(first, conflicting)).unique(
            subset=[date_col, id_col, precip_col],
            keep='first',
            maintain_order=True,
        )
        raise ValueError(
            f"Duplicate date and id values with differing precipitation "
            f"values found:\n{dup_df}",
        )
    df_raw = df_raw.filter(first)
    # now round the date to the defined frequency by aggregating to the specified
    # frequency. An interval without any observation stays missing, an interval a
    # station did not report is not the same as a reported zero
    interval = _interval(freq)
    observed = _valid(precip_col, df_raw.schema[precip_col])
    df = (
        df_raw.select(
            pl.col(id_col),
            pl.col(date_col).dt.truncate(interval),
            pl.col(precip_col),
        )
        .group_by(id_col, date_col)
        .agg(
            pl.when(observed.len() > 0)
            .then(observed.sum())
            .otherwise(None)
            .alias(precip_col),
        )
    )
    # a gapless time series covering the whole period, the bounds are taken from
    # the column itself so that the range keeps its time unit and time zone
    full_time = df.select(
        pl.datetime_range(
            start=pl.col(date_col).min(),
            end=pl.col(date_col).max(),
            interval=interval,
        ).alias(date_col),
    )
    # every station over the full time series, missing combinations become null
    full_grid = df.select(
        pl.col(id_col).unique(maintain_order=True),
    ).join(full_time, how='cross')
    df = full_grid.join(df, on=[id_col, date_col], how='left')
    # sort to ensure proper order
    df = df.sort(by=[id_col, date_col])
    if kept:
        # every station of the result has a row in the constants, so this only
        # broadcasts the values and never introduces a missing value
        df = df.join(constants, on=id_col, how='left', validate='m:1')
    return df


# WGS84, the ellipsoid geopy measures the geodesic distances on
_WGS84_A = 6378137.0
_WGS84_E2 = (1 / 298.257223563) * (2 - 1 / 298.257223563)


def _chord_distances(
        lat: NDArray[np.float64],
        lon: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The straight line distances through the earth between all stations.

    The straight line between two points is never longer than a path along the
    surface, so this is a lower bound of the geodesic distance and can be used to
    rule out pairs of stations without computing the expensive exact distance.
    Over 10 km the two differ by about a millimeter.
    """
    phi = np.radians(lat)
    lam = np.radians(lon)
    sin_phi = np.sin(phi)
    # the distance from the point to the polar axis and its height
    n = _WGS84_A / np.sqrt(1 - _WGS84_E2 * sin_phi * sin_phi)
    xy = n * np.cos(phi)
    coords = np.column_stack((xy * np.cos(lam), xy * np.sin(lam), n * (1 - _WGS84_E2) * sin_phi))  # noqa: E501
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))


def _geodesic_chunk(
        args: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """The exact geodesic distance in meters of every given pair of stations."""
    lat1, lon1, lat2, lon2 = args
    return np.array([
        geodesic((a, b), (c, d)).meters
        for a, b, c, d in zip(lat1, lon1, lat2, lon2)
    ])


def find_station_neighbors(
        station_metadata: Any,
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
        and longitude, either a :class:`pandas.DataFrame` or a
        :class:`polars.DataFrame`.
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
    missing_cols = required_cols - set(_frames.columns(station_metadata))

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if d < 0:
        raise ValueError('d must be non-negative')

    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError('max_neighbors must be at least 1')

    ids = _frames.values(station_metadata, id_col)
    lat = _frames.floats(station_metadata, lat_col)
    lon = _frames.floats(station_metadata, lon_col)
    # remove the rows with missing coordinates and the repeated stations
    known = ~(np.isnan(lat) | np.isnan(lon))
    _, first = np.unique(ids[known], return_index=True)
    keep = np.flatnonzero(known)[np.sort(first)]
    ids, lat, lon = ids[keep], lat[keep], lon[keep]

    n = ids.size
    if n < 2:
        return {station_id: () for station_id in ids}

    # only the pairs that could possibly be within d are measured exactly, the
    # geodesic distance of a pair is never shorter than the straight line
    dist_matrix = np.full((n, n), np.inf)
    np.fill_diagonal(dist_matrix, 0.0)
    rows, cols = np.nonzero(np.triu(_chord_distances(lat, lon) <= d, k=1))

    if rows.size:
        workers = _workers(n_jobs, rows.size)
        # a chunk per worker, the geodesic of a single pair is far too little
        # work to hand to another process on its own
        chunks = [
            (lat[r], lon[r], lat[c], lon[c])
            for r, c in zip(
                np.array_split(rows, workers), np.array_split(cols, workers),
            )
        ]
        if workers == 1:
            measured = [_geodesic_chunk(chunk) for chunk in chunks]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                measured = list(executor.map(_geodesic_chunk, chunks))
        distances = np.concatenate(measured)
        dist_matrix[rows, cols] = distances
        dist_matrix[cols, rows] = distances

    result: dict[int, tuple[int, ...]] = {}

    for i in range(n):
        distances = dist_matrix[i]
        order = np.argsort(distances, kind='stable')
        within = order[(distances[order] > 0) & (distances[order] <= d)]
        if max_neighbors is not None:
            within = within[:max_neighbors]
        result[ids[i]] = tuple(ids[j] for j in within)

    return result


def _positions(
        lookup: NDArray[np.generic],
        values: NDArray[np.generic],
) -> NDArray[np.intp]:
    """Locate ``values`` in the ascending ``lookup``, -1 where they are missing."""
    if lookup.size == 0:
        return np.full(values.shape, -1, dtype=np.intp)
    positions = np.searchsorted(lookup, values)
    clipped = np.clip(positions, 0, lookup.size - 1)
    found = lookup[clipped] == values
    return np.where(found, clipped, -1).astype(np.intp)


class _Layout(NamedTuple):
    """How the rows of a long format frame map onto a ``(time x station)`` matrix.

    ``rows`` and ``cols`` are the row and column index of every row of the frame.
    They are ``None`` for the common case of a frame that already is a complete
    grid ordered by station and then by time, as
    :func:`pwsqc.prepare_timeseries` builds it -- reshaping is then all it takes
    and the index arrays are not needed.
    """
    times: NDArray[Any]
    station_ids: NDArray[Any]
    rows: NDArray[np.intp] | None
    cols: NDArray[np.intp] | None

    @property
    def shape(self) -> tuple[int, int]:
        return self.times.size, self.station_ids.size


def _ascending(values: NDArray[np.generic]) -> bool:
    """Whether ``values`` is strictly ascending."""
    if values.size < 2:
        return True
    steps = np.diff(values)
    # a zero of the same dtype, so that a time unit is kept
    return bool((steps > np.zeros((), dtype=steps.dtype)).all())


def _grid_layout(
        ids: NDArray[np.generic],
        dates: NDArray[np.generic],
) -> _Layout | None:
    """Recognize a frame that is a complete grid ordered by station and time.

    :return: The layout, or ``None`` if the rows are not such a grid and have to
        be mapped one by one.
    """
    n = ids.size
    if n == 0:
        return None
    # the stations have to come in equally sized consecutive blocks
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    n_stations = boundaries.size + 1
    if n % n_stations:
        return None
    n_times = n // n_stations
    if not np.array_equal(boundaries, np.arange(1, n_stations) * n_times):
        return None
    station_ids = ids[::n_times]
    if not _ascending(station_ids):
        return None
    # ... every one of them covering the same ascending timestamps
    times = dates[:n_times]
    if not _ascending(times):
        return None
    if n_stations > 1 and not bool(
            (dates.reshape(n_stations, n_times) == times).all(),
    ):
        return None
    return _Layout(times=times, station_ids=station_ids, rows=None, cols=None)


def _layout(
        data: Any,
        id_col: str,
        date_col: str,
) -> _Layout:
    """Determine how the rows of ``data`` map onto a ``(time x station)`` matrix."""
    ids = _frames.values(data, id_col)
    dates = _frames.values(data, date_col)

    layout = _grid_layout(ids, dates)
    if layout is None:
        times = np.unique(dates)
        station_ids = np.unique(ids)
        rows = _positions(times, dates)
        cols = _positions(station_ids, ids)
        # a duplicate would silently overwrite a value while reshaping
        cells = rows.astype(np.int64) * station_ids.size + cols
        if np.unique(cells).size != cells.size:
            raise ValueError(
                f'duplicate ({date_col}, {id_col}) combinations are not allowed, '
                f'use prepare_timeseries to build a regular time series',
            )
        layout = _Layout(
            times=times, station_ids=station_ids, rows=rows, cols=cols,
        )

    if layout.times.size > 1:
        steps = np.unique(np.diff(layout.times))
        if steps.size != 1:
            raise ValueError(
                f'the time series is not regular, found {len(steps)} different '
                f'time steps, use prepare_timeseries to build a regular time series',
            )
    return layout


def _to_wide(
        data: Any,
        id_col: str,
        date_col: str,
        value_cols: tuple[str, ...],
) -> tuple[list[NDArray[np.float64]], _Layout]:
    """Reshape the long format data into ``(time x station)`` matrices.

    The filters operate on regular time series where consecutive rows are
    consecutive measurement intervals, so the time axis is validated to be
    strictly increasing with a constant step.

    :param data: Long format DataFrame with one row per station and time step.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param value_cols: Columns to reshape, one matrix is returned per column.
    :return: A list of ``(n_times, n_stations)`` matrices and the layout holding
        the shared time axis and the station ids of the matrix columns.
    """
    missing_cols = ({id_col, date_col} | set(value_cols)) - set(
        _frames.columns(data),
    )
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    layout = _layout(data=data, id_col=id_col, date_col=date_col)
    n_times, n_stations = layout.shape

    matrices = []
    for col in value_cols:
        column = _frames.floats(data, col)
        if layout.rows is None:
            # the values are already grouped by station, transposing the blocks
            # is the whole reshape
            matrix = np.ascontiguousarray(
                column.reshape(n_stations, n_times).T,
            )
        else:
            # combinations without a row stay NaN, as a pivot would leave them
            matrix = np.full((n_times, n_stations), np.nan)
            matrix[layout.rows, layout.cols] = column
        matrices.append(matrix)
    return matrices, layout


def _to_long(
        layout: _Layout,
        values: NDArray[_ScalarT],
) -> NDArray[_ScalarT]:
    """Map a ``(time x station)`` matrix back onto the rows of the frame."""
    if layout.rows is None:
        return values.T.ravel()
    return values[layout.rows, layout.cols]


def _neighbor_index(
        neighbors: Mapping[int, Sequence[int]],
        station_ids: NDArray[np.int64],
) -> list[NDArray[np.intp]]:
    """Translate the neighbor ids into column indices of the wide matrices.

    Neighbors without observations in the data set are silently dropped, they
    cannot contribute to the median or to the correlations either way.
    """
    index = []
    for station_id in station_ids:
        ids = np.asarray(neighbors.get(station_id, ()))
        positions = (
            _positions(station_ids, ids)
            if len(ids) else np.array([], dtype=np.intp)
        )
        index.append(positions[positions >= 0].astype(np.intp))
    return index


def _neighbor_stats(
        values: NDArray[np.float64],
        neighbor_index: list[NDArray[np.intp]],
        n_jobs: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Compute the median and the number of reporting neighbors per interval.

    :param values: ``(n_times, n_stations)`` matrix of rainfall observations.
    :param neighbor_index: Column indices of the neighbors of each station.
    :param n_jobs: Number of threads to spread the stations over. Defaults to
        os.cpu_count().
    :return: The ``(n_times, n_stations)`` median of the neighboring stations and
        the number of neighboring stations reporting an observation.
    """
    med = np.full(values.shape, np.nan)
    cnt = np.zeros(values.shape, dtype=np.int64)

    def _station(i: int) -> None:
        columns = neighbor_index[i]
        if len(columns) == 0:
            return
        med[:, i], cnt[:, i] = _row_nanmedian(values[:, columns])

    _run(_station, range(len(neighbor_index)), n_jobs)
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
