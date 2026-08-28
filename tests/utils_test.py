import numpy as np
import pandas as pd
import pytest
from pwsqc import Config
from pwsqc import find_station_neighbors
from pwsqc import prepare_timeseries
from pwsqc.utils import _compute_row_distances
from pwsqc.utils import _neighbor_index
from pwsqc.utils import _neighbor_stats
from pwsqc.utils import _rle
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


def test_prepare_timeseries_aggregates_and_fills_gaps() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1, 1, 2],
        'date': pd.to_datetime(
            [
                '2024-05-01 00:01', '2024-05-01 00:03',  # same 5 min interval
                '2024-05-01 00:17',                      # a gap in between
                '2024-05-01 00:02',
            ], utc=True,
        ),
        'precip': [0.1, 0.2, 0.5, 0.0],
    })
    result = prepare_timeseries(df_raw)
    assert list(result.columns) == ['intern_id', 'date', 'precip']
    station = result[result['intern_id'] == 1].sort_values('date')
    # the two observations of the first interval are summed up
    assert station['precip'].tolist()[0] == pytest.approx(0.30000000000000004)
    # the intervals in between exist and are NaN for the station without data
    assert len(station) == 4
    assert np.isnan(station['precip'].tolist()[1])
    assert station['precip'].tolist()[3] == 0.5
    # both stations cover the same time range
    assert result['intern_id'].value_counts().unique().tolist() == [4]


def test_prepare_timeseries_carries_additional_columns_along() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1, 2],
        'date': pd.to_datetime(
            [
                '2024-05-01 00:01',
                '2024-05-01 00:17',
                '2024-05-01 00:02',
            ],
            utc=True,
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
    station = result[result['intern_id'] == 1]
    assert len(station) == 4
    assert station['geometry'].tolist() == ['POINT(1 1)'] * 4
    assert station['city_id'].tolist() == [7] * 4
    assert result[result['intern_id'] == 2]['city_id'].tolist() == [9] * 4


def test_prepare_timeseries_keeps_the_selected_columns_only() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 2],
        'date': pd.to_datetime(['2024-05-01 00:01'] * 2, utc=True),
        'precip': [0.1, 0.0],
        'geometry': ['POINT(1 1)', 'POINT(2 2)'],
        'city_id': [7, 9],
    })
    result = prepare_timeseries(df_raw, keep_cols=('geometry',))
    assert list(result.columns) == ['intern_id', 'date', 'precip', 'geometry']
    result = prepare_timeseries(df_raw, keep_cols=())
    assert list(result.columns) == ['intern_id', 'date', 'precip']


def test_prepare_timeseries_fills_in_a_partly_missing_column() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1, 2],
        'date': pd.to_datetime(
            [
                '2024-05-01 00:01',
                '2024-05-01 00:07',
                '2024-05-01 00:02',
            ],
            utc=True,
        ),
        'precip': [0.1, 0.2, 0.0],
        'alt': [12.0, np.nan, np.nan],
    })
    result = prepare_timeseries(df_raw)
    # the one value a station reported is used for all of its intervals
    assert result[result['intern_id'] == 1]['alt'].tolist() == [12.0, 12.0]
    assert np.isnan(result[result['intern_id'] == 2]['alt']).all()


def test_prepare_timeseries_rejects_columns_that_vary_within_a_station() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1],
        'date': pd.to_datetime(
            ['2024-05-01 00:01', '2024-05-01 00:07'], utc=True,
        ),
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


def test_prepare_timeseries_unknown_keep_cols() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1],
        'date': pd.to_datetime(['2024-05-01 00:01'], utc=True),
        'precip': [0.1],
    })
    with pytest.raises(ValueError) as exc_info:
        prepare_timeseries(df_raw, keep_cols=('geometry',))
    assert str(exc_info.value) == "Missing required columns: ['geometry']"


def test_prepare_timeseries_ignores_exact_duplicates() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1],
        'date': pd.to_datetime(['2024-05-01 00:01'] * 2, utc=True),
        'precip': [0.1, 0.1],
    })
    assert prepare_timeseries(df_raw)['precip'].tolist() == [0.1]


def test_prepare_timeseries_rejects_conflicting_duplicates() -> None:
    df_raw = pd.DataFrame({
        'intern_id': [1, 1],
        'date': pd.to_datetime(['2024-05-01 00:01'] * 2, utc=True),
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


def test_compute_row_distances() -> None:
    coords = [(52.0, 4.9), (52.009, 4.9), (52.018, 4.9)]
    i, row = _compute_row_distances((0, coords))
    assert i == 0
    # the distances to the stations before it are left to the caller
    assert np.isnan(row[0])
    assert row[1] == pytest.approx(1000, abs=10)
    assert row[2] == pytest.approx(2000, abs=20)


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


def test_to_wide_duplicate_rows(make_data) -> None:
    data = make_data({1: [0.0, 0.0]})
    with pytest.raises(ValueError) as exc_info:
        _to_wide(pd.concat([data, data]), 'intern_id', 'date', ('precip',))
    assert 'duplicate (date, intern_id) combinations' in str(exc_info.value)


def test_to_wide_irregular_time_series(make_data) -> None:
    data = make_data({1: [0.0, 0.0, 0.0, 0.0]})
    with pytest.raises(ValueError) as exc_info:
        _to_wide(data.drop(index=1), 'intern_id', 'date', ('precip',))
    assert 'the time series is not regular' in str(exc_info.value)


def test_to_wide_single_interval(make_data) -> None:
    (values,), times, ids = _to_wide(
        make_data({1: [0.5], 2: [0.0]}), 'intern_id', 'date', ('precip',),
    )
    assert values.tolist() == [[0.5, 0.0]]
    assert len(times) == 1
    assert ids.tolist() == [1, 2]


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
