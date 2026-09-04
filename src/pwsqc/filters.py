from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _frames
from ._frames import FrameT
from .utils import _neighbor_index
from .utils import _neighbor_stats
from .utils import _rle
from .utils import _row_nanmedian
from .utils import _run
from .utils import _to_long
from .utils import _to_wide

NO_FLAG = 0
FLAG = 1
NOT_ENOUGH_INFO = -1


def _faulty_zero_flags(
        precip: NDArray[np.float64],
        ref: NDArray[np.float64],
        n_int: int,
) -> NDArray[np.int8]:
    """Attribute the FZ flags of a single station.

    :param precip: Rainfall time series of the station.
    :param ref: Binary reference of the surrounding area, 1 for wet, 0 for dry and
        NaN where it could not be determined.
    :param n_int: Number of intervals the station must report zero rainfall while
        the surrounding area is wet.
    """
    flags = np.zeros(precip.size, dtype=np.int8)
    # binary time series of the station, 1 for wet and 0 for dry observations
    wet_dry = np.where(precip > 0, 1.0, precip)
    starts, ends, values = _rle(wet_dry)
    # dry periods as measured by the station, a missing value interrupts them
    dry = (ends - starts + 1 > n_int) & (values == 0)

    for start, end in zip(starts[dry], ends[dry]):
        window = ref[start:end + 1]
        # only relevant if the surrounding area was wet for more than n_int
        # intervals during the dry period of the station
        if np.count_nonzero(window == 1) <= n_int:
            continue
        # ... and only if those wet intervals were consecutive
        wet_starts, wet_ends, wet_values = _rle(window)
        wet = (wet_ends - wet_starts + 1 > n_int) & (wet_values == 1)
        if not wet.any():
            continue

        # the interval where the previous n_int intervals were dry at the station
        # and wet in the median of the neighbors
        flags[start + wet_starts[wet][0] + n_int:end + 1] = FLAG
        # once flagged, the flagging continues until the station reports rainfall
        # again, missing values are ignored
        i = end + 1
        while i < wet_dry.size and (np.isnan(wet_dry[i]) or wet_dry[i] == 0):
            flags[i] = FLAG
            i += 1

    return flags


def faulty_zero_filter(
        data: FrameT,
        neighbors: Mapping[int, Sequence[int]],
        n_stat: int = 5,
        n_int: int = 6,
        id_col: str = 'intern_id',
        date_col: str = 'date',
        precip_col: str = 'precip',
        flag_col: str = 'FZflag',
        n_jobs: int | None = None,
) -> FrameT:
    """Apply the Faulty Zero (FZ) filter to the precipitation data.

    A station is compared with the median of its neighboring stations. The flag is
    set to 1 once the station reported zero rainfall for more than ``n_int``
    consecutive intervals while the median of the neighbors was larger than zero
    during more than ``n_int`` consecutive intervals of that period. The flagging
    continues until the station reports nonzero rainfall again. The flag is -1
    whenever fewer than ``n_stat`` neighboring stations report an observation.

    :param data: Long format DataFrame with a regular time series per station, as
        returned by :func:`pwsqc.prepare_timeseries`. Either a
        :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param neighbors: Mapping of a station id to the ids of its neighbors, as
        returned by :func:`pwsqc.find_station_neighbors`.
    :param n_stat: Minimum number of neighboring stations with an observation.
    :param n_int: Number of intervals a station has to report zero rainfall while
        the surrounding area is wet.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param precip_col: Column name holding the rainfall of the interval in mm.
    :param flag_col: Name of the column the flags are written to.
    :param n_jobs: Number of threads to spread the stations over. Defaults to
        os.cpu_count().
    :return: DataFrame with an additional ``FZflag`` column indicating flagged data
        points.
    """
    (values,), layout = _to_wide(
        data=data,
        id_col=id_col,
        date_col=date_col,
        value_cols=(precip_col,),
    )
    station_ids = layout.station_ids
    neighbor_index = _neighbor_index(neighbors=neighbors, station_ids=station_ids)
    med, cnt = _neighbor_stats(
        values=values, neighbor_index=neighbor_index, n_jobs=n_jobs,
    )

    flags = np.zeros(values.shape, dtype=np.int8)

    def _station(i: int) -> None:
        # a station without any observation or with too few neighbors cannot be
        # evaluated at all
        if (
                len(neighbors.get(station_ids[i], ())) < n_stat or
                np.isnan(values[:, i]).all()
        ):
            flags[:, i] = NOT_ENOUGH_INFO
            return

        # binary reference of the surrounding area, NaN where the median could not
        # be constructed from at least n_stat stations
        ref = np.where(med[:, i] > 0, 1.0, np.where(med[:, i] == 0, 0.0, np.nan))
        ref[cnt[:, i] < n_stat] = np.nan
        flags[:, i] = _faulty_zero_flags(
            precip=values[:, i],
            ref=ref,
            n_int=n_int,
        )
        # if too few neighbors have observations the flag cannot be attributed
        flags[cnt[:, i] < n_stat, i] = NOT_ENOUGH_INFO

    _run(_station, range(station_ids.size), n_jobs)

    return _frames.with_columns(
        data, {flag_col: _to_long(layout, flags).astype(np.int8)},
    )


def high_influx_filter(
        data: FrameT,
        neighbors: Mapping[int, Sequence[int]],
        n_stat: int = 5,
        phi_a: float = 0.4,
        phi_b: float = 10,
        id_col: str = 'intern_id',
        date_col: str = 'date',
        precip_col: str = 'precip',
        flag_col: str = 'HIflag',
        n_jobs: int | None = None,
) -> FrameT:
    """Apply the High Influx (HI) filter to the precipitation data.

    A rainfall measurement that is significantly larger than the median of the
    surrounding stations is flagged. If that median is lower than ``phi_a``, the
    observation is flagged when it is larger than ``phi_b``. If the median is equal
    to or larger than ``phi_a``, the observation is flagged when it is larger than
    ``median * phi_b / phi_a``. The flag is -1 whenever fewer than ``n_stat``
    neighboring stations report an observation.

    :param data: Long format DataFrame with a regular time series per station, as
        returned by :func:`pwsqc.prepare_timeseries`. Either a
        :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param neighbors: Mapping of a station id to the ids of its neighbors, as
        returned by :func:`pwsqc.find_station_neighbors`.
    :param n_stat: Minimum number of neighboring stations with an observation.
    :param phi_a: Rainfall threshold of the median of the neighbors in mm.
    :param phi_b: Rainfall threshold of the station itself in mm.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param precip_col: Column name holding the rainfall of the interval in mm.
    :param flag_col: Name of the column the flags are written to.
    :param n_jobs: Number of threads to spread the stations over. Defaults to
        os.cpu_count().
    :return: DataFrame with an additional ``HIflag`` column indicating flagged data
        points.
    """
    (values,), layout = _to_wide(
        data=data,
        id_col=id_col,
        date_col=date_col,
        value_cols=(precip_col,),
    )
    station_ids = layout.station_ids
    neighbor_index = _neighbor_index(neighbors=neighbors, station_ids=station_ids)
    med, cnt = _neighbor_stats(
        values=values, neighbor_index=neighbor_index, n_jobs=n_jobs,
    )

    # comparisons with NaN are always False, so intervals without an observation
    # of the station itself or without a median are never flagged
    with np.errstate(invalid='ignore'):
        flags = np.where(
            ((values > phi_b) & (med < phi_a)) |
            ((med >= phi_a) & (values > phi_b * med / phi_a)),
            FLAG,
            NO_FLAG,
        ).astype(np.int8)
    flags[cnt < n_stat] = NOT_ENOUGH_INFO

    # a station without any observation or with too few neighbors cannot be
    # evaluated at all
    n_neighbors = np.array(
        [len(neighbors.get(station_id, ())) for station_id in station_ids],
    )
    unusable = (n_neighbors < n_stat) | np.isnan(values).all(axis=0)
    flags[:, unusable] = NOT_ENOUGH_INFO

    return _frames.with_columns(
        data, {flag_col: _to_long(layout, flags).astype(np.int8)},
    )


def _compare_start(
        precip: NDArray[np.float64],
        m_int: int,
        m_rain: int,
) -> NDArray[np.int64]:
    """Determine the first interval of the comparison window of every interval.

    The window covers the previous ``m_int`` intervals, or any longer period so
    that it contains at least ``m_rain`` intervals with nonzero rainfall.

    :param precip: Rainfall time series of the station.
    :param m_int: Number of intervals of the comparison window.
    :param m_rain: Number of nonzero rainfall intervals the window must contain.
    :return: The index of the first interval of the window, -1 where no window
        could be constructed.
    """
    n = precip.size
    start = np.full(n, -1, dtype=np.int64)

    # window that reaches back until m_rain rainy intervals are covered
    rain_rows = np.flatnonzero(precip > 0)
    n_rain = np.cumsum(precip > 0)
    if m_rain <= 0:
        # no rainfall required, the window may start at the interval itself
        by_rain = np.arange(n, dtype=np.int64)
    else:
        by_rain = np.full(n, -1, dtype=np.int64)
        enough_rain = n_rain >= m_rain
        # the m_rain-th last rainy interval, and one interval before it
        by_rain[enough_rain] = rain_rows[n_rain[enough_rain] - m_rain] - 1

    # window that reaches back m_int intervals
    by_int = np.arange(n) - m_int + 1

    # both criteria must be met, the longer of the two windows is used
    usable = (by_rain >= 0) & (by_int >= 0)
    start[usable] = np.minimum(by_rain[usable], by_int[usable])
    return start


def _prefix(
        values: NDArray[Any],
        dtype: type[np.generic],
) -> NDArray[Any]:
    """Cumulative sums of the columns, with a row of zeros in front of them.

    The result is allocated in one piece and the cumulative sum is written into
    it directly, so that the large matrices are not copied again to prepend the
    zeros. Both sides are laid out column by column, so that the sum of a column
    runs along contiguous memory instead of jumping a row at every step, which
    is what makes the difference for the extended precision sums.
    """
    out = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=dtype, order='F')
    np.cumsum(np.asfortranarray(values), axis=0, dtype=dtype, out=out[1:])
    return out


def _neighbor_correlation_bias(
        precip: NDArray[np.float64],
        neighbor_values: NDArray[np.float64],
        start: NDArray[np.int64],
        n_stat: int,
        m_match: int,
        dbc: float,
        chunk_size: int = 4096,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Correlate a station with its neighbors over the comparison window.

    Both metrics are computed from cumulative sums, so the cost does not depend on
    the length of the comparison window.

    :param precip: Bias corrected rainfall time series of the station.
    :param neighbor_values: ``(n_times, n_neighbors)`` matrix of the bias corrected
        rainfall of the neighboring stations.
    :param start: First interval of the comparison window of every interval.
    :param n_stat: Minimum number of neighboring stations the metrics are needed of.
    :param m_match: Minimum number of overlapping intervals of a neighbor.
    :param dbc: The default bias correction factor.
    :return: The median correlation and the median relative bias of every interval,
        NaN where they could not be determined.
    """
    n = precip.size
    med_cor = np.full(n, np.nan)
    med_bias = np.full(n, np.nan)

    # a neighbor only contributes where both stations have an observation, the
    # neighbor values are NaN wherever the station itself is NaN already
    overlap = ~np.isnan(neighbor_values)
    x = np.where(overlap, neighbor_values, 0.0)
    y = np.where(overlap, precip[:, None], 0.0)

    # the cumulative sums of the values are accumulated with extended precision,
    # so that the window sums stay accurate over very long time series. The
    # overlapping intervals are counted, whole numbers that int64 holds exactly
    # and accumulates an order of magnitude faster
    c_n = _prefix(overlap, np.int64)
    c_x = _prefix(x, np.longdouble)
    c_y = _prefix(y, np.longdouble)
    c_xx = _prefix(x * x, np.longdouble)
    c_yy = _prefix(y * y, np.longdouble)
    c_xy = _prefix(x * y, np.longdouble)
    del overlap, x, y

    rows = np.flatnonzero(start >= 0)
    for offset in range(0, rows.size, chunk_size):
        chunk = rows[offset:offset + chunk_size]
        lo = start[chunk]
        ends: Any
        if chunk[-1] - chunk[0] + 1 == chunk.size:
            # the ends of the windows of a chunk are usually consecutive, then
            # they can be sliced out instead of gathered row by row
            ends = np.s_[chunk[0] + 1:chunk[-1] + 2]
        else:
            ends = chunk + 1
        count = (c_n[ends] - c_n[lo]).astype(np.float64)
        sum_x = (c_x[ends] - c_x[lo]).astype(np.float64)
        sum_y = (c_y[ends] - c_y[lo]).astype(np.float64)
        sum_xx = (c_xx[ends] - c_xx[lo]).astype(np.float64)
        sum_yy = (c_yy[ends] - c_yy[lo]).astype(np.float64)
        sum_xy = (c_xy[ends] - c_xy[lo]).astype(np.float64)

        # only neighbors with enough overlapping intervals are considered and
        # only if there are enough of those neighbors
        selected = count > m_match
        selected[selected.sum(axis=1) < n_stat] = False

        with np.errstate(invalid='ignore', divide='ignore'):
            cov = sum_xy - sum_x * sum_y / count
            var_x = sum_xx - sum_x * sum_x / count
            var_y = sum_yy - sum_y * sum_y / count
            # the correlation is undefined if either time series is constant
            cor = np.where(
                (var_x > 0) & (var_y > 0),
                cov / np.sqrt(var_x * var_y),
                np.nan,
            )
            # relative bias of the raw station observations against the bias
            # corrected observations of the neighbor
            bias = (sum_y / dbc - sum_x) / sum_x

        cor = np.where(selected, cor, np.nan)
        bias = np.where(selected, bias, np.nan)
        # the flag depends on the number of neighbors the correlation is known of
        cor_median, cor_count = _row_nanmedian(cor)
        enough = cor_count >= n_stat
        med_cor[chunk[enough]] = cor_median[enough]
        bias_median, bias_count = _row_nanmedian(bias)
        enough_bias = bias_count > 0
        med_bias[chunk[enough_bias]] = bias_median[enough_bias]

    return med_cor, med_bias


def station_outlier_filter(
        data: FrameT,
        neighbors: Mapping[int, Sequence[int]],
        n_stat: int = 5,
        m_int: int = 4032,
        m_rain: int = 100,
        m_match: int = 200,
        gamma: float = 0.15,
        dbc: float = 1.24,
        id_col: str = 'intern_id',
        date_col: str = 'date',
        precip_col: str = 'precip',
        fz_col: str = 'FZflag',
        hi_col: str = 'HIflag',
        flag_col: str = 'SOflag',
        bias_col: str = 'bias',
        n_jobs: int | None = 1,
) -> FrameT:
    """Apply the Station Outlier (SO) filter to the precipitation data.

    A station is an outlier when it shows very different rainfall dynamics than its
    neighbors. Every interval is compared with the neighboring stations over a
    previous period of ``m_int`` intervals, or any longer period containing at
    least ``m_rain`` intervals with nonzero rainfall. Only neighbors with more than
    ``m_match`` overlapping intervals are considered. The flag is set to 1 if the
    median of the Pearson correlations falls short of ``gamma`` and to -1 if the
    correlation is known of fewer than ``n_stat`` neighboring stations.

    The intervals flagged by the FZ and the HI filter are excluded from the
    comparison, so both filters have to be applied first.

    The median relative bias of the same comparison is written to the ``bias``
    column, it is the input of :func:`pwsqc.bias_correction`.

    :param data: Long format DataFrame with a regular time series per station and
        the flags of the FZ and the HI filter. Either a
        :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param neighbors: Mapping of a station id to the ids of its neighbors, as
        returned by :func:`pwsqc.find_station_neighbors`.
    :param n_stat: Minimum number of neighboring stations to compare with.
    :param m_int: Number of intervals of the comparison window.
    :param m_rain: Number of nonzero rainfall intervals the window must contain.
    :param m_match: Minimum number of overlapping intervals of a neighbor.
    :param gamma: Threshold of the median correlation with the neighbors.
    :param dbc: The default bias correction factor. It does not affect the flags,
        but the relative bias is computed against the corrected neighbors.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param precip_col: Column name holding the rainfall of the interval in mm.
    :param fz_col: Column name holding the flags of the FZ filter.
    :param hi_col: Column name holding the flags of the HI filter.
    :param flag_col: Name of the column the flags are written to.
    :param bias_col: Name of the column the median relative bias is written to.
    :param n_jobs: Number of threads to spread the stations over. Defaults to
        one: the comparison window sums of a station are far larger than the
        caches, so this loop is limited by the memory bandwidth rather than by
        the cores and more threads buy little while every one of them holds
        another set of those sums.
    :return: DataFrame with an additional ``SOflag`` column indicating flagged data
        points and a ``bias`` column with the median relative bias.
    """
    (values, fz, hi), layout = _to_wide(
        data=data,
        id_col=id_col,
        date_col=date_col,
        value_cols=(precip_col, fz_col, hi_col),
    )
    station_ids = layout.station_ids
    neighbor_index = _neighbor_index(neighbors=neighbors, station_ids=station_ids)

    # the outlier is determined on the intervals that were not flagged already,
    # the correction factor is constant and does not affect the correlation.
    # Every station is a column here and the whole comparison runs along it, so
    # the working set is laid out column by column: the time series of a station
    # and of its neighbors are then contiguous all the way into the window sums
    shape = values.shape
    corrected = np.asfortranarray(values * dbc)
    corrected[(fz == FLAG) | (hi == FLAG)] = np.nan
    # the matrices are large for long time series, release them right away
    del values, fz, hi

    flags = np.zeros(shape, dtype=np.int8, order='F')
    bias = np.full(shape, np.nan, order='F')

    def _station(i: int) -> None:
        precip = corrected[:, i]
        # a station without any observation or with too few neighbors cannot be
        # evaluated at all
        if (
                len(neighbors.get(station_ids[i], ())) < n_stat or
                np.isnan(precip).all()
        ):
            flags[:, i] = NOT_ENOUGH_INFO
            return

        # indexing the columns copies them out already, this only keeps the
        # column by column layout of the copy
        neighbor_values = np.asfortranarray(corrected[:, neighbor_index[i]])
        # intervals without an observation of the station itself do not overlap
        neighbor_values[np.isnan(precip)] = np.nan

        med_cor, med_bias = _neighbor_correlation_bias(
            precip=precip,
            neighbor_values=neighbor_values,
            start=_compare_start(precip=precip, m_int=m_int, m_rain=m_rain),
            n_stat=n_stat,
            m_match=m_match,
            dbc=dbc,
        )
        with np.errstate(invalid='ignore'):
            flags[:, i] = np.where(med_cor < gamma, FLAG, NO_FLAG)
        flags[np.isnan(med_cor), i] = NOT_ENOUGH_INFO
        bias[:, i] = med_bias

    _run(_station, range(station_ids.size), n_jobs)

    return _frames.with_columns(
        data,
        {
            flag_col: _to_long(layout, flags).astype(np.int8),
            bias_col: _to_long(layout, bias),
        },
    )


def _bias_correction_factors(
        bias: NDArray[np.float64],
        flags: NDArray[np.int8],
        dbc: float,
        beta: float,
) -> NDArray[np.float64]:
    """Build the bias correction factor time series of a single station.

    :param bias: Median relative bias with the neighboring stations.
    :param flags: Flags of the SO filter.
    :param dbc: The default bias correction factor.
    :param beta: Relative threshold a change of the factor has to exceed.
    """
    factors = np.full(bias.size, float(dbc))
    # only intervals where the station is known not to be an outlier are used
    candidates = np.flatnonzero(flags == NO_FLAG)
    if candidates.size == 0:
        return factors

    with np.errstate(divide='ignore', invalid='ignore'):
        new_factors = 1 / (1 + bias[candidates])

    current = float(dbc)
    offset = 0
    while offset < candidates.size:
        # a change is only systematic if abs(log(new / previous)) > log(1 + beta).
        # The equivalent form without the logarithm is much better conditioned,
        # the two are only compared for a systematic change after all. An
        # undefined new factor never triggers a change.
        candidate = new_factors[offset:]
        changed = np.flatnonzero(
            (candidate > current * (1 + beta)) | (candidate * (1 + beta) < current),
        )
        if changed.size == 0:
            break
        offset += changed[0]
        current = float(new_factors[offset])
        factors[candidates[offset] + 1:] = current
        offset += 1

    return factors


def bias_correction(
        data: FrameT,
        dbc: float = 1.24,
        beta: float = 0.2,
        id_col: str = 'intern_id',
        date_col: str = 'date',
        so_col: str = 'SOflag',
        bias_col: str = 'bias',
        bcf_col: str = 'BCF',
        n_jobs: int | None = None,
) -> FrameT:
    """Compute the bias correction factor of every station and interval.

    Every station starts out with the default bias correction factor ``dbc``.
    Whenever the station is known not to be an outlier, a new factor is derived
    from the median relative bias with the neighboring stations. If the new factor
    deviates from the current one by more than a factor of ``1 + beta``, the change
    is deemed systematic and the new factor is used from the next interval on.

    The bias corrected rainfall of an interval is ``precip * BCF``.

    :param data: Long format DataFrame with the ``SOflag`` and ``bias`` columns of
        :func:`pwsqc.station_outlier_filter`. Either a
        :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param dbc: The default bias correction factor.
    :param beta: Relative threshold a change of the factor has to exceed.
    :param id_col: Column name for station identifier.
    :param date_col: Column name holding the (interval end) timestamps.
    :param so_col: Column name holding the flags of the SO filter.
    :param bias_col: Column name holding the median relative bias.
    :param bcf_col: Name of the column the correction factors are written to.
    :param n_jobs: Number of threads to spread the stations over. Defaults to
        os.cpu_count().
    :return: DataFrame with an additional ``BCF`` column.
    """
    (bias, flags), layout = _to_wide(
        data=data,
        id_col=id_col,
        date_col=date_col,
        value_cols=(bias_col, so_col),
    )

    factors = np.empty(bias.shape)

    def _station(i: int) -> None:
        factors[:, i] = _bias_correction_factors(
            bias=bias[:, i],
            flags=flags[:, i].astype(np.int8),
            dbc=dbc,
            beta=beta,
        )

    _run(_station, range(layout.station_ids.size), n_jobs)

    return _frames.with_columns(data, {bcf_col: _to_long(layout, factors)})


def apply_flags(
        data: FrameT,
        strict: bool = False,
        precip_col: str = 'precip',
        flag_cols: tuple[str, ...] = ('FZflag', 'HIflag', 'SOflag'),
        bcf_col: str | None = 'BCF',
        out_col: str = 'precip_qc',
) -> FrameT:
    """Apply the flags and the bias correction to the rainfall observations.

    :param data: Long format DataFrame with the flag columns of the filters.
        Either a :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    :param strict: If ``False`` only the intervals with a flag of 1 are discarded
        ("filtered flex"), if ``True`` the intervals without enough information to
        determine the flag are discarded as well ("filtered strict").
    :param precip_col: Column name holding the rainfall of the interval in mm.
    :param flag_cols: Column names of the flags to apply.
    :param bcf_col: Column name of the bias correction factor, ``None`` to not
        apply a bias correction.
    :param out_col: Name of the column the result is written to.
    :return: DataFrame with an additional ``precip_qc`` column where the flagged
        intervals are missing.
    """
    missing_cols = set(flag_cols) - set(_frames.columns(data))
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    precip = _frames.floats(data, precip_col)
    if bcf_col is not None:
        precip = precip * _frames.floats(data, bcf_col)

    discard = np.zeros(_frames.n_rows(data), dtype=bool)
    for col in flag_cols:
        flags = _frames.values(data, col)
        discard |= flags == FLAG
        if strict:
            discard |= flags == NOT_ENOUGH_INFO

    return _frames.with_columns(data, {out_col: np.where(discard, np.nan, precip)})
