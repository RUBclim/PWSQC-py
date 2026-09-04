import numpy as np
import pandas as pd
import pytest
from pwsqc import Config
from pwsqc import find_station_neighbors
from pwsqc import prepare_timeseries
from pwsqc.utils import _chord_distances
from pwsqc.utils import _duplicate_runs
from pwsqc.utils import _geodesic_chunk
from pwsqc.utils import _grid_layout
from pwsqc.utils import _neighbor_index
from pwsqc.utils import _neighbor_stats
from pwsqc.utils import _positions
from pwsqc.utils import _rle
from pwsqc.utils import _row_nanmedian
from pwsqc.utils import _to_long
from pwsqc.utils import _to_wide


def test_config_defaults_are_the_values_of_the_paper() -> None:
    config = Config()
    assert config.d == 10_000
    assert config.n_stat == 5
    assert config.n_int == 6
    assert config.phi_a == 0.4
    assert config.phi_b == 10
    assert config.m_int == 4032
    assert config.m_rain == 100
    assert config.m_match == 200
    assert config.gamma == 0.15
    assert config.beta == 0.2
    assert config.dbc == 1.24


def test_prepare_timeseries_aggregates_and_fills_gaps(
        frame, dates, select, values_of,
) -> None:
    df_raw = frame({
        'intern_id': [1, 1, 1, 2],
        'date': dates(
            '2024-05-01 00:01', '2024-05-01 00:03',  # same 5 min interval
            '2024-05-01 00:17',                      # a gap in between
            '2024-05-01 00:02',
        ),
        'precip': [0.1, 0.2, 0.5, 0.0],
    })
    result = prepare_timeseries(df_raw)
    assert list(result.columns) == ['intern_id', 'date', 'precip']
    station = select(result, 1)
    precip = values_of(station, 'precip')
    # the two observations of the first interval are summed up
    assert precip[0] == pytest.approx(0.30000000000000004)
    # the intervals in between exist and are missing for the station without data
    assert len(station) == 4
    assert precip[1] is None
    assert precip[3] == 0.5
    # both stations cover the same time range
    assert len(select(result, 2)) == 4


def test_prepare_timeseries_carries_additional_columns_along(
        frame, dates, select, values_of,
) -> None:
    df_raw = frame({
        'intern_id': [1, 1, 2],
        'date': dates(
            '2024-05-01 00:01',
            '2024-05-01 00:17',
            '2024-05-01 00:02',
        ),
        'precip': [0.1, 0.5, 0.0],
        'geometry': ['POINT(1 1)', 'POINT(1 1)', 'POINT(2 2)'],
        'city_id': [7, 7, 9],
    })
    result = prepare_timeseries(df_raw)
    # the additional columns keep their place and their dtype
    assert list(result.columns) == [
        'intern_id', 'date', 'precip', 'geometry', 'city_id',
    ]
    assert result['city_id'].dtype == df_raw['city_id'].dtype
    # the intervals that were added by the resampling get them as well
    station = select(result, 1)
    assert len(station) == 4
    assert values_of(station, 'geometry') == ['POINT(1 1)'] * 4
    assert values_of(station, 'city_id') == [7] * 4
    assert values_of(select(result, 2), 'city_id') == [9] * 4


def test_prepare_timeseries_keeps_the_selected_columns_only(frame, dates) -> None:
    df_raw = frame({
        'intern_id': [1, 2],
        'date': dates(*['2024-05-01 00:01'] * 2),
        'precip': [0.1, 0.0],
        'geometry': ['POINT(1 1)', 'POINT(2 2)'],
        'city_id': [7, 9],
    })
    result = prepare_timeseries(df_raw, keep_cols=('geometry',))
    assert list(result.columns) == ['intern_id', 'date', 'precip', 'geometry']
    result = prepare_timeseries(df_raw, keep_cols=())
    assert list(result.columns) == ['intern_id', 'date', 'precip']


def test_prepare_timeseries_fills_in_a_partly_missing_column(
        frame, dates, select, values_of,
) -> None:
    df_raw = frame({
        'intern_id': [1, 1, 2],
        'date': dates(
            '2024-05-01 00:01',
            '2024-05-01 00:07',
            '2024-05-01 00:02',
        ),
        'precip': [0.1, 0.2, 0.0],
        'alt': [12.0, np.nan, np.nan],
    })
    result = prepare_timeseries(df_raw)
    # the one value a station reported is used for all of its intervals
    assert values_of(select(result, 1), 'alt') == [12.0, 12.0]
    assert values_of(select(result, 2), 'alt') == [None, None]


def test_prepare_timeseries_rejects_columns_that_vary_within_a_station(
        frame, dates,
) -> None:
    df_raw = frame({
        'intern_id': [1, 1],
        'date': dates('2024-05-01 00:01', '2024-05-01 00:07'),
        'precip': [0.1, 0.2],
        'temperature': [17.0, 18.0],
    })
    with pytest.raises(ValueError) as exc_info:
        prepare_timeseries(df_raw)
    assert "the columns ['temperature'] are not constant" in str(exc_info.value)
    # ... also when they were asked for explicitly
    with pytest.raises(ValueError):
        prepare_timeseries(df_raw, keep_cols=('temperature',))
    # ... but they can be excluded
    result = prepare_timeseries(df_raw, keep_cols=())
    assert list(result.columns) == ['intern_id', 'date', 'precip']


def test_prepare_timeseries_unknown_keep_cols(frame, dates) -> None:
    df_raw = frame({
        'intern_id': [1],
        'date': dates('2024-05-01 00:01'),
        'precip': [0.1],
    })
    with pytest.raises(ValueError) as exc_info:
        prepare_timeseries(df_raw, keep_cols=('geometry',))
    assert str(exc_info.value) == "Missing required columns: ['geometry']"


def test_prepare_timeseries_ignores_exact_duplicates(
        frame, dates, values_of,
) -> None:
    df_raw = frame({
        'intern_id': [1, 1],
        'date': dates(*['2024-05-01 00:01'] * 2),
        'precip': [0.1, 0.1],
    })
    assert values_of(prepare_timeseries(df_raw), 'precip') == [0.1]


def test_prepare_timeseries_rejects_conflicting_duplicates(frame, dates) -> None:
    df_raw = frame({
        'intern_id': [1, 1],
        'date': dates(*['2024-05-01 00:01'] * 2),
        'precip': [0.1, 0.2],
    })
    with pytest.raises(ValueError) as exc_info:
        prepare_timeseries(df_raw)
    assert 'Duplicate date and id values' in str(exc_info.value)


META = pd.DataFrame({
    'intern_id': [1, 2, 3, 4, 5],
    # roughly 1 km apart each, the last one is far away
    'lat': [52.0, 52.009, 52.018, 52.027, 53.0],
    'lon': [4.9, 4.9, 4.9, 4.9, 4.9],
})


def test_find_station_neighbors_sorted_by_distance() -> None:
    neighbors = find_station_neighbors(META, d=2500, n_jobs=1)
    assert neighbors[1] == (2, 3)
    assert neighbors[2] == (1, 3, 4)
    assert neighbors[5] == ()


def test_find_station_neighbors_defaults_to_all_cpus() -> None:
    assert find_station_neighbors(META, d=2500)[1] == (2, 3)


def test_geodesic_chunk() -> None:
    lat = np.array([52.0, 52.0])
    lon = np.array([4.9, 4.9])
    distances = _geodesic_chunk(
        (lat, lon, np.array([52.009, 52.018]), np.array([4.9, 4.9])),
    )
    assert distances[0] == pytest.approx(1000, abs=10)
    assert distances[1] == pytest.approx(2000, abs=20)


def test_chord_distances_never_exceed_the_geodesic() -> None:
    lat = np.array([52.0, 52.009, 52.018, 53.0])
    lon = np.array([4.9, 4.9, 5.1, 6.0])
    chord = _chord_distances(lat, lon)
    assert np.diag(chord).tolist() == [0.0] * 4
    for i in range(4):
        for j in range(4):
            exact = _geodesic_chunk(
                (lat[i:i + 1], lon[i:i + 1], lat[j:j + 1], lon[j:j + 1]),
            )[0]
            # the straight line is the lower bound the pre-filter relies on and
            # it is tight enough to hardly ever admit a pair that is too far
            assert chord[i, j] <= exact + 1e-6
            # and it is tight over the ranges the filters use
            if exact < 20_000:
                assert exact - chord[i, j] < 0.01


def test_find_station_neighbors_max_neighbors() -> None:
    neighbors = find_station_neighbors(META, d=100_000, max_neighbors=2, n_jobs=1)
    assert neighbors[1] == (2, 3)


def test_find_station_neighbors_excludes_colocated_stations() -> None:
    meta = pd.DataFrame({
        'intern_id': [1, 2, 3],
        'lat': [52.0, 52.0, 52.009],
        'lon': [4.9, 4.9, 4.9],
    })
    # station 2 reports the exact same location as station 1
    assert find_station_neighbors(meta, d=2500, n_jobs=1)[1] == (3,)


def test_find_station_neighbors_ignores_stations_without_coordinates() -> None:
    meta = pd.DataFrame({
        'intern_id': [1, 2, 3],
        'lat': [52.0, np.nan, 52.009],
        'lon': [4.9, 4.9, 4.9],
    })
    neighbors = find_station_neighbors(meta, d=2500, n_jobs=1)
    assert set(neighbors) == {1, 3}


def test_find_station_neighbors_single_station() -> None:
    meta = pd.DataFrame({'intern_id': [1], 'lat': [52.0], 'lon': [4.9]})
    assert find_station_neighbors(meta, n_jobs=1) == {1: ()}


def test_find_station_neighbors_missing_columns() -> None:
    with pytest.raises(ValueError) as exc_info:
        find_station_neighbors(pd.DataFrame({'intern_id': [1]}))
    assert 'Missing required columns' in str(exc_info.value)


def test_find_station_neighbors_negative_range() -> None:
    with pytest.raises(ValueError) as exc_info:
        find_station_neighbors(META, d=-1)
    assert str(exc_info.value) == 'd must be non-negative'


def test_find_station_neighbors_invalid_max_neighbors() -> None:
    with pytest.raises(ValueError) as exc_info:
        find_station_neighbors(META, max_neighbors=0)
    assert str(exc_info.value) == 'max_neighbors must be at least 1'


def test_to_wide_missing_columns(make_data) -> None:
    data = make_data({1: [0.0, 0.0]})
    with pytest.raises(ValueError) as exc_info:
        _to_wide(data, 'intern_id', 'date', ('nope',))
    assert str(exc_info.value) == "Missing required columns: ['nope']"


def test_to_wide_duplicate_rows(make_data, concat) -> None:
    data = make_data({1: [0.0, 0.0]})
    with pytest.raises(ValueError) as exc_info:
        _to_wide(concat(data, data), 'intern_id', 'date', ('precip',))
    assert 'duplicate (date, intern_id) combinations' in str(exc_info.value)


def test_to_wide_irregular_time_series(make_data, drop_row) -> None:
    data = make_data({1: [0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError) as exc_info:
        _to_wide(drop_row(data, 1), 'intern_id', 'date', ('precip',))
    assert 'the time series is not regular' in str(exc_info.value)


def test_to_wide_single_interval(make_data) -> None:
    (values,), layout = _to_wide(
        make_data({1: [0.5], 2: [0.0]}), 'intern_id', 'date', ('precip',),
    )
    assert values.tolist() == [[0.5, 0.0]]
    assert len(layout.times) == 1
    assert layout.station_ids.tolist() == [1, 2]


def test_neighbor_index_skips_unknown_stations() -> None:
    ids = np.array([1, 2, 3])
    # station 3 is not in the mapping at all, 99 has no observations
    index = _neighbor_index({1: (2, 99), 2: ()}, ids)
    assert [i.tolist() for i in index] == [[1], [], []]


def test_neighbor_stats_without_neighbors() -> None:
    values = np.array([[1.0, 2.0], [np.nan, np.nan]])
    med, cnt = _neighbor_stats(values, [np.array([1]), np.array([], dtype=int)])
    assert med[0, 0] == 2.0
    # the second station has no neighbors, the second interval has no observation
    assert np.isnan(med[0, 1])
    assert np.isnan(med[1]).all()
    assert cnt.tolist() == [[1, 0], [0, 0]]


def test_rle_empty() -> None:
    starts, ends, values = _rle(np.array([], dtype=np.float64))
    assert starts.tolist() == ends.tolist() == values.tolist() == []


def test_rle_treats_every_nan_as_its_own_run() -> None:
    starts, ends, values = _rle(np.array([0.0, 0.0, np.nan, np.nan, 0.0, 1.0]))
    assert starts.tolist() == [0, 2, 3, 4, 5]
    assert ends.tolist() == [1, 2, 3, 4, 5]
    assert values[0] == 0.0
    assert np.isnan(values[1]) and np.isnan(values[2])
    assert values[3] == 0.0
    assert values[4] == 1.0


def test_row_nanmedian_without_any_column() -> None:
    median, count = _row_nanmedian(np.empty((3, 0)))
    assert np.isnan(median).all()
    assert count.tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    'values',
    (
        pytest.param(
            np.array([[1.0, 2.0, np.nan], [np.nan, np.nan, np.nan]]),
            id='a row without any value',
        ),
        pytest.param(
            np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]),
            id='an even and an odd number of values',
        ),
        pytest.param(
            np.array([[np.nan, 1.0, 2.0, 8.0], [3.0, 4.0, 5.0, 6.0]]),
            id='four values',
        ),
    ),
)
def test_row_nanmedian_matches_numpy(values) -> None:
    median, count = _row_nanmedian(values)
    with np.errstate(invalid='ignore'):
        expected = np.nanmedian(
            np.where(np.isnan(values).all(axis=1)[:, None], 0.0, values), axis=1,
        )
    expected = np.where(np.isnan(values).all(axis=1), np.nan, expected)
    assert np.array_equal(median, expected, equal_nan=True)
    assert count.tolist() == np.count_nonzero(~np.isnan(values), axis=1).tolist()


def test_duplicate_runs_of_an_empty_frame() -> None:
    empty = np.array([])
    first, conflicting = _duplicate_runs(empty, empty, empty)
    assert first.tolist() == conflicting.tolist() == []


def test_duplicate_runs_of_a_column_without_missing_values() -> None:
    # an integer column can never hold a NaN, the extra check is skipped
    first, conflicting = _duplicate_runs(
        dates=np.array([1, 1, 1, 2]),
        ids=np.array([1, 1, 2, 2]),
        values=np.array([7, 7, 8, 9]),
    )
    assert first.tolist() == [True, False, True, True]
    assert conflicting.tolist() == [False, False, False, False]


def test_duplicate_runs_treats_two_missing_values_as_equal() -> None:
    first, conflicting = _duplicate_runs(
        dates=np.array([1, 1, 1]),
        ids=np.array([1, 1, 1]),
        values=np.array([np.nan, np.nan, 1.0]),
    )
    assert first.tolist() == [True, False, False]
    # the two missing values are duplicates, the number is a conflict
    assert conflicting.tolist() == [False, False, True]


def test_positions_of_an_empty_lookup() -> None:
    found = _positions(np.array([], dtype=np.int64), np.array([1, 2]))
    assert found.tolist() == [-1, -1]


@pytest.mark.parametrize(
    ('ids', 'dates'),
    (
        pytest.param([1, 1, 2], [0, 1, 0], id='the blocks do not divide the rows'),
        pytest.param([1, 1, 1, 2], [0, 1, 2, 0], id='the blocks differ in size'),
        pytest.param([2, 2, 1, 1], [0, 1, 0, 1], id='the stations are not ascending'),
        pytest.param([1, 1], [1, 0], id='the timestamps are not ascending'),
        pytest.param([1, 1, 2, 2], [0, 1, 0, 2], id='the blocks cover other times'),
    ),
)
def test_grid_layout_only_accepts_a_complete_grid(ids, dates) -> None:
    assert _grid_layout(np.array(ids), np.array(dates)) is None


def test_grid_layout_of_an_empty_frame() -> None:
    assert _grid_layout(np.array([]), np.array([])) is None


def test_to_wide_of_a_frame_that_is_not_ordered_by_station(make_data, frame) -> None:
    """The reshaping does not depend on the order of the rows."""
    ordered = make_data({1: [0.5, 1.5], 2: [0.0, 2.0]})
    (expected,), by_station = _to_wide(ordered, 'intern_id', 'date', ('precip',))
    assert by_station.rows is None

    # the very same rows, but ordered by time and then by station
    shuffled = frame({
        'intern_id': [1, 2, 1, 2],
        'date': _repeat_dates(ordered),
        'precip': [0.5, 0.0, 1.5, 2.0],
    })
    (values,), layout = _to_wide(shuffled, 'intern_id', 'date', ('precip',))
    # this one has to be mapped row by row
    assert layout.rows is not None
    assert np.array_equal(values, expected)
    assert layout.station_ids.tolist() == by_station.station_ids.tolist()
    # ... and mapping it back gets the rows of that frame, not of the other one
    assert _to_long(layout, values).tolist() == [0.5, 0.0, 1.5, 2.0]
    assert _to_long(by_station, expected).tolist() == [0.5, 1.5, 0.0, 2.0]


def _repeat_dates(ordered):
    """The two timestamps of ``ordered``, each of them twice."""
    stamps = sorted(set(_column(ordered, 'date')))
    return [stamps[0], stamps[0], stamps[1], stamps[1]]


def _column(data, name):
    if type(data).__module__.split('.')[0] == 'polars':
        return data[name].to_list()
    return data[name].tolist()


def test_find_station_neighbors_without_any_pair_in_range() -> None:
    # no pair of stations is within a meter of each other
    assert find_station_neighbors(META, d=1, n_jobs=1) == {
        1: (), 2: (), 3: (), 4: (), 5: (),
    }
