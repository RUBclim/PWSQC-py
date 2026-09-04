from typing import Any

import numpy as np
import pandas as pd
import pytest
from pwsqc import apply_flags
from pwsqc import bias_correction
from pwsqc import faulty_zero_filter
from pwsqc import high_influx_filter
from pwsqc import station_outlier_filter
from pwsqc.filters import _compare_start
from pwsqc.filters import _neighbor_correlation_bias

NAN = float('nan')


# # # FZ filter # # #

def test_faulty_zero_flag_starts_after_n_int_dry_intervals(make_data, flags_of) -> None:
    #                   0    1    2    3    4    5    6    7    8    9
    data = make_data({
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        2: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=2)
    # the station is dry from interval 1 on, the area is wet from interval 1 on,
    # so the flag starts n_int intervals later and lasts until it rains again
    assert flags_of(result, 1, 'FZflag') == [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    # the input is not modified
    assert 'FZflag' not in data.columns


def test_faulty_zero_flag_continues_through_missing_values(make_data, flags_of) -> None:
    data = make_data({
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, NAN, NAN, 0.0, 0.0],
        2: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        3: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=2)
    # the dry period of the station ends at the missing values, but once the
    # station is flagged the flagging continues until it reports rainfall again
    assert flags_of(result, 1, 'FZflag') == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


def test_faulty_zero_needs_consecutive_wet_intervals(make_data, flags_of) -> None:
    data = make_data({
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        2: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=2)
    # showers passing by are no reason to distrust a station reporting zeroes
    assert flags_of(result, 1, 'FZflag') == [0] * 10


def test_faulty_zero_without_enough_wet_intervals(make_data, flags_of) -> None:
    data = make_data({
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        2: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        3: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=2)
    assert flags_of(result, 1, 'FZflag') == [0] * 10


def test_faulty_zero_without_enough_reporting_neighbors(
        make_data,
        flags_of,
) -> None:
    data = make_data({
        1: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
        2: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 1.0, 1.0, NAN, NAN, 1.0, 1.0, 0.0, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=2)
    flags = flags_of(result, 1, 'FZflag')
    # the median cannot be built from n_stat stations in those two intervals
    assert flags[4] == flags[5] == -1


def test_faulty_zero_station_that_cannot_be_evaluated(make_data, flags_of) -> None:
    data = make_data({
        1: [0.0, 0.0, 0.0],
        2: [1.0, 1.0, 1.0],
        3: [NAN, NAN, NAN],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = faulty_zero_filter(data, neighbors, n_stat=2, n_int=1)
    # station 3 has no observation at all, station 4 is not a neighbor of anyone
    assert flags_of(result, 3, 'FZflag') == [-1, -1, -1]
    # station 1 has two neighbors, but only one of them reports
    assert flags_of(result, 1, 'FZflag') == [-1, -1, -1]


def test_faulty_zero_station_without_enough_neighbors(make_data, flags_of) -> None:
    data = make_data({1: [0.0, 0.0], 2: [1.0, 1.0], 3: [1.0, 1.0]})
    result = faulty_zero_filter(data, {1: (2,), 2: (1, 3), 3: (1, 2)}, n_stat=2)
    assert flags_of(result, 1, 'FZflag') == [-1, -1]


# # # HI filter # # #

@pytest.mark.parametrize(
    ('station', 'neighbor', 'expected'),
    (
        pytest.param(11.0, 0.0, 1, id='dry surroundings, above phi_b'),
        pytest.param(9.0, 0.0, 0, id='dry surroundings, below phi_b'),
        pytest.param(11.0, 0.5, 0, id='wet surroundings, below the ratio'),
        pytest.param(13.0, 0.5, 1, id='wet surroundings, above the ratio'),
        pytest.param(12.5, 0.5, 0, id='wet surroundings, exactly at the ratio'),
        pytest.param(NAN, 0.5, 0, id='no observation'),
    ),
)
def test_high_influx_thresholds(
        make_data,
        flags_of,
        station,
        neighbor,
        expected,
) -> None:
    data = make_data({
        1: [station, 0.0],
        2: [neighbor, 0.0],
        3: [neighbor, 0.0],
    })
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = high_influx_filter(data, neighbors, n_stat=2, phi_a=0.4, phi_b=10)
    assert flags_of(result, 1, 'HIflag')[0] == expected
    assert 'HIflag' not in data.columns


def test_high_influx_without_enough_reporting_neighbors(make_data, flags_of) -> None:
    data = make_data({1: [25.0, 25.0], 2: [0.0, NAN], 3: [0.0, 0.0]})
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    result = high_influx_filter(data, neighbors, n_stat=2)
    assert flags_of(result, 1, 'HIflag') == [1, -1]


def test_high_influx_station_that_cannot_be_evaluated(make_data, flags_of) -> None:
    data = make_data({1: [NAN, NAN], 2: [0.0, 0.0], 3: [0.0, 0.0]})
    result = high_influx_filter(data, {1: (2,), 2: (1, 3), 3: (1, 2)}, n_stat=2)
    # too few neighbors for station 1, no observations at all for station 1
    assert flags_of(result, 1, 'HIflag') == [-1, -1]


# # # SO filter # # #

def _outlier_data(n_times: int = 60, seed: int = 3) -> dict[int, list[float]]:
    """Three well behaved stations, one that measures something else entirely."""
    rng = np.random.default_rng(seed)
    field = rng.gamma(1.0, 0.4, n_times) * (rng.random(n_times) < 0.5)
    factors = (1.0, 0.9, 1.1, 1.05)
    series = {
        i: (field * factor).tolist()
        for i, factor in enumerate(factors, start=1)
    }
    # station 5 is at the wrong location, its rainfall is unrelated
    other = rng.gamma(1.0, 0.4, n_times) * (rng.random(n_times) < 0.5)
    series[5] = other.tolist()
    return series


def test_station_outlier_flags_the_station_with_other_dynamics(
        make_data,
        flags_of,
) -> None:
    data = make_data(_outlier_data())
    neighbors = {i: tuple(j for j in range(1, 6) if j != i) for i in range(1, 6)}
    data = faulty_zero_filter(data, neighbors, n_stat=3, n_int=6)
    data = high_influx_filter(data, neighbors, n_stat=3)
    result = station_outlier_filter(
        data, neighbors, n_stat=3, m_int=20, m_rain=5, m_match=10, gamma=0.5,
    )
    correlated = flags_of(result, 2, 'SOflag')
    outlier = flags_of(result, 5, 'SOflag')
    # no comparison window is available at the start of the time series
    assert correlated[0] == outlier[0] == -1
    assert correlated[-1] == 0
    assert outlier[-1] == 1
    assert 'SOflag' not in data.columns


def _biased_network(make_data: Any, factor: float) -> Any:
    """One station and three neighbors that all measure ``factor`` times as much."""
    rng = np.random.default_rng(7)
    field = rng.gamma(1.0, 0.4, 60) * (rng.random(60) < 0.5)
    series = {1: field.tolist()}
    series.update({i: (field * factor).tolist() for i in (2, 3, 4)})
    data = make_data(series)
    neighbors = {i: tuple(j for j in range(1, 5) if j != i) for i in range(1, 5)}
    data = faulty_zero_filter(data, neighbors, n_stat=3, n_int=6)
    data = high_influx_filter(data, neighbors, n_stat=3)
    return data, neighbors


@pytest.mark.parametrize(
    ('factor', 'dbc', 'expected'),
    (
        pytest.param(1.0, 1.0, 0.0, id='no bias'),
        # the station measures 1 / 0.8 times as much as its neighbors
        pytest.param(0.8, 1.0, 0.25, id='neighbors measure less'),
        # the default correction lifts the neighbors to the level of the station
        pytest.param(0.8, 1.25, 0.0, id='corrected away by the default'),
    ),
)
def test_station_outlier_relative_bias(
        make_data,
        flags_of,
        factor,
        dbc,
        expected,
) -> None:
    data, neighbors = _biased_network(make_data, factor)
    result = station_outlier_filter(
        data, neighbors, n_stat=3, m_int=20, m_rain=5, m_match=10, gamma=0.5,
        dbc=dbc,
    )
    assert flags_of(result, 1, 'bias')[-1] == pytest.approx(expected, abs=1e-12)
    # identical dynamics are never an outlier
    assert flags_of(result, 1, 'SOflag')[-1] == 0


def test_station_outlier_without_enough_neighbors(make_data, flags_of) -> None:
    data = make_data({1: [0.0, 1.0], 2: [0.0, 1.0], 3: [0.0, 1.0]})
    neighbors = {1: (2,), 2: (1, 3), 3: (1, 2)}
    data = faulty_zero_filter(data, neighbors, n_stat=2)
    data = high_influx_filter(data, neighbors, n_stat=2)
    result = station_outlier_filter(data, neighbors, n_stat=2, m_int=1, m_rain=1)
    assert flags_of(result, 1, 'SOflag') == [-1, -1]
    assert flags_of(result, 1, 'bias') == [None, None]


def test_station_outlier_of_a_station_without_observations(make_data, flags_of) -> None:
    data = make_data({1: [NAN, NAN], 2: [0.0, 1.0], 3: [0.0, 1.0]})
    neighbors = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    data = faulty_zero_filter(data, neighbors, n_stat=2)
    data = high_influx_filter(data, neighbors, n_stat=2)
    result = station_outlier_filter(data, neighbors, n_stat=2, m_int=1, m_rain=1)
    assert flags_of(result, 1, 'SOflag') == [-1, -1]


def test_station_outlier_excludes_the_intervals_of_the_other_filters(
        make_data,
        set_flag,
) -> None:
    data = make_data(_outlier_data())
    neighbors = {i: tuple(j for j in range(1, 6) if j != i) for i in range(1, 6)}
    data = faulty_zero_filter(data, neighbors, n_stat=3, n_int=6)
    data = high_influx_filter(data, neighbors, n_stat=3)
    flagged = set_flag(data, station_id=1, column='HIflag', value=1)
    plain = station_outlier_filter(
        data, neighbors, n_stat=3, m_int=20, m_rain=5, m_match=10, gamma=0.5,
    )
    masked = station_outlier_filter(
        flagged, neighbors, n_stat=3, m_int=20, m_rain=5, m_match=10, gamma=0.5,
    )
    # with every interval of station 1 discarded it cannot serve as a neighbor
    assert (plain['SOflag'] != masked['SOflag']).any()


@pytest.mark.parametrize(
    ('precip', 'm_int', 'm_rain', 'expected'),
    (
        pytest.param([1.0] * 5, 3, 2, [-1, -1, 0, 1, 2], id='rain in every interval'),
        pytest.param([1.0] * 5, 3, 9, [-1] * 5, id='never enough rain'),
        pytest.param([0.0] * 5, 3, 0, [-1, -1, 0, 1, 2], id='no rain required'),
        pytest.param(
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 2, 2, [-1, -1, -1, -1, -1, 0],
            id='the window grows until it covers m_rain rainy intervals',
        ),
        pytest.param(
            [1.0, 0.0, 0.0, 0.0, 1.0], 2, 2, [-1] * 5,
            id='no interval left before the first rainy one',
        ),
    ),
)
def test_compare_start(precip, m_int, m_rain, expected) -> None:
    start = _compare_start(np.array(precip), m_int=m_int, m_rain=m_rain)
    assert start.tolist() == expected


# # # bias correction # # #

def _bias_frame(
        bias: list[float],
        so_flag: list[int],
        station_id: int = 1,
) -> pd.DataFrame:
    times = pd.date_range('2024-05-01 00:05', periods=len(bias), freq='5min', tz='UTC')
    return pd.DataFrame({
        'intern_id': station_id,
        'date': times,
        'precip': 1.0,
        'bias': bias,
        'SOflag': so_flag,
    })


def test_bias_correction_keeps_the_default_without_information(flags_of) -> None:
    data = _bias_frame([NAN] * 4, [-1] * 4)
    result = bias_correction(data, dbc=1.24, beta=0.2)
    assert flags_of(result, 1, 'BCF') == [1.24] * 4


def test_bias_correction_ignores_small_changes(flags_of) -> None:
    # 1 / (1 + 0.1) = 0.909, which is within a factor of 1.2 of 1.0
    data = _bias_frame([0.1] * 4, [0] * 4)
    result = bias_correction(data, dbc=1.0, beta=0.2)
    assert flags_of(result, 1, 'BCF') == [1.0] * 4


def test_bias_correction_applies_a_systematic_change(flags_of) -> None:
    # 1 / (1 - 0.5) = 2.0, twice the previous factor
    data = _bias_frame([NAN, -0.5, -0.5, -0.5], [-1, 0, 0, 0])
    result = bias_correction(data, dbc=1.0, beta=0.2)
    # the new factor is used from the next interval on, never retroactively
    assert flags_of(result, 1, 'BCF') == [1.0, 1.0, 2.0, 2.0]


def test_bias_correction_only_uses_intervals_without_an_outlier_flag(flags_of) -> None:
    data = _bias_frame([-0.5, -0.5, -0.5, -0.5], [1, 1, 0, 1])
    result = bias_correction(data, dbc=1.0, beta=0.2)
    assert flags_of(result, 1, 'BCF') == [1.0, 1.0, 1.0, 2.0]


def test_bias_correction_of_the_last_interval_is_not_retroactive(flags_of) -> None:
    data = _bias_frame([NAN, NAN, NAN, -0.5], [-1, -1, -1, 0])
    result = bias_correction(data, dbc=1.0, beta=0.2)
    # there is no interval left the new factor could be applied to
    assert flags_of(result, 1, 'BCF') == [1.0] * 4


def test_bias_correction_updates_repeatedly(flags_of) -> None:
    data = _bias_frame([-0.5, -0.75, 0.0, 0.0], [0, 0, 0, 0])
    result = bias_correction(data, dbc=1.0, beta=0.2)
    # 1 -> 2 -> 4, the third value of 1.0 differs by more than the threshold again
    assert flags_of(result, 1, 'BCF') == [1.0, 2.0, 4.0, 1.0]


# # # applying the flags # # #

def _flagged_frame() -> pd.DataFrame:
    times = pd.date_range('2024-05-01 00:05', periods=4, freq='5min', tz='UTC')
    return pd.DataFrame({
        'intern_id': 1,
        'date': times,
        'precip': [1.0, 2.0, 3.0, 4.0],
        'FZflag': [0, 1, 0, 0],
        'HIflag': [0, 0, -1, 0],
        'SOflag': [0, 0, 0, 0],
        'BCF': [1.0, 1.0, 1.0, 2.0],
    })


def test_apply_flags_flex() -> None:
    result = apply_flags(_flagged_frame())
    assert result['precip_qc'].tolist()[0] == 1.0
    assert np.isnan(result['precip_qc'].tolist()[1])
    # a flag of -1 is kept when filtering flex
    assert result['precip_qc'].tolist()[2] == 3.0
    # the bias correction is applied
    assert result['precip_qc'].tolist()[3] == 8.0


def test_apply_flags_strict() -> None:
    result = apply_flags(_flagged_frame(), strict=True)
    assert np.isnan(result['precip_qc'].tolist()[1])
    assert np.isnan(result['precip_qc'].tolist()[2])


def test_apply_flags_without_bias_correction() -> None:
    result = apply_flags(_flagged_frame(), bcf_col=None)
    assert result['precip_qc'].tolist()[3] == 4.0


def test_apply_flags_missing_column() -> None:
    with pytest.raises(ValueError) as exc_info:
        apply_flags(_flagged_frame().drop(columns='SOflag'))
    assert str(exc_info.value) == "Missing required columns: ['SOflag']"


def test_neighbor_correlation_bias_with_gaps_between_the_windows() -> None:
    """The intervals a window could be built for need not be consecutive."""
    rng = np.random.default_rng(0)
    precip = rng.random(12)
    neighbor_values = rng.random((12, 3))
    kwargs: dict[str, Any] = {
        'precip': precip,
        'neighbor_values': neighbor_values,
        'n_stat': 2,
        'm_match': 1,
        'dbc': 1.24,
    }
    consecutive = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    with_gaps = consecutive.copy()
    with_gaps[[3, 5, 8]] = -1

    everything = _neighbor_correlation_bias(start=consecutive, **kwargs)
    some = _neighbor_correlation_bias(start=with_gaps, **kwargs)
    kept = with_gaps >= 0
    for whole, part in zip(everything, some):
        # the intervals that were left out have no result ...
        assert np.isnan(part[~kept]).all()
        # ... and the others are the ones they are without the gaps
        assert np.array_equal(whole[kept], part[kept], equal_nan=True)
