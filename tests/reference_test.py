"""Validate the implementation against the R reference implementation.

The expected values in ``testing/reference`` were produced by the original
R code of de Vos et al. (2019), see ``testing/reference/generate.py``.
"""
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pwsqc import bias_correction
from pwsqc import faulty_zero_filter
from pwsqc import high_influx_filter
from pwsqc import station_outlier_filter

from tests.conftest import BACKENDS

REFERENCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'testing', 'reference',
)
# the parameters the reference output was produced with
D = 10_000
N_STAT = 5
N_INT = 6
PHI_A = 0.4
PHI_B = 10
M_INT = 100
M_RAIN = 20
M_MATCH = 30
GAMMA = 0.15
DBC = 1.24
BETA = 0.2


def _expected(name: str) -> np.ndarray:
    path = os.path.join(REFERENCE, f'{name}.csv.gz')
    return pd.read_csv(path, float_precision='round_trip').to_numpy(dtype=np.float64)


@pytest.fixture(scope='module', params=BACKENDS)
def reference_data(request) -> tuple[Any, dict[int, tuple[int, ...]], list[int]]:
    wide = pd.read_csv(
        os.path.join(REFERENCE, 'Ndataset.csv.gz'),
        index_col=0,
        parse_dates=[0],
        float_precision='round_trip',
    )
    station_ids = [int(c) for c in wide.columns]
    wide.columns = station_ids
    wide.index = pd.DatetimeIndex(wide.index)
    long = wide.melt(ignore_index=False, var_name='intern_id', value_name='precip')
    data = long.rename_axis('date').reset_index()[['intern_id', 'date', 'precip']]
    if request.param == 'polars':
        import polars as pl

        data = pl.from_pandas(data)

    neighbor_table = pd.read_csv(
        os.path.join(REFERENCE, 'neighbourlist.csv'),
        dtype=str,
    )
    neighbors = {}
    for _, row in neighbor_table.iterrows():
        ids = row['neighbours']
        neighbors[int(row['station_id'])] = (
            tuple(int(i) for i in ids.split(',')) if isinstance(ids, str) else ()
        )
    return data, neighbors, station_ids


@pytest.fixture(scope='module')
def qc_result(
        reference_data: tuple[Any, dict[int, tuple[int, ...]], list[int]],
) -> Any:
    data, neighbors, _ = reference_data
    result = faulty_zero_filter(
        data,
        neighbors=neighbors,
        n_stat=N_STAT,
        n_int=N_INT,
    )
    result = high_influx_filter(
        result,
        neighbors=neighbors,
        n_stat=N_STAT,
        phi_a=PHI_A,
        phi_b=PHI_B,
    )
    result = station_outlier_filter(
        result,
        neighbors=neighbors,
        n_stat=N_STAT,
        m_int=M_INT,
        m_rain=M_RAIN,
        m_match=M_MATCH,
        gamma=GAMMA,
        dbc=DBC,
    )
    return bias_correction(result, dbc=DBC, beta=BETA)


def _as_matrix(
        result: Any,
        column: str,
        station_ids: list[int],
) -> np.ndarray:
    if type(result).__module__.split('.')[0] == 'polars':
        import polars as pl

        wide = result.pivot(on='intern_id', index='date', values=column)
        return wide.select(
            pl.col(str(i)).cast(pl.Float64) for i in station_ids
        ).to_numpy()
    wide = result.pivot(index='date', columns='intern_id', values=column)
    return wide[station_ids].to_numpy(dtype=np.float64)


@pytest.mark.parametrize(
    ('column', 'expected_name'),
    (
        ('FZflag', 'FZ_flags'),
        ('HIflag', 'HI_flags'),
        ('SOflag', 'SO_flags'),
    ),
)
def test_flags_match_the_reference_implementation(
        qc_result: Any,
        reference_data: tuple[Any, dict[int, tuple[int, ...]], list[int]],
        column: str,
        expected_name: str,
) -> None:
    _, _, station_ids = reference_data
    expected = _expected(expected_name)
    got = _as_matrix(qc_result, column, station_ids)
    # the reference data set contains every flag value
    assert set(np.unique(expected)) == {-1, 0, 1}
    assert np.array_equal(got, expected)


def test_bias_correction_matches_the_reference_implementation(
        qc_result: Any,
        reference_data: tuple[Any, dict[int, tuple[int, ...]], list[int]],
) -> None:
    _, _, station_ids = reference_data
    expected = _expected('BCF')
    got = _as_matrix(qc_result, 'BCF', station_ids)
    # the correction factor is actually adjusted over time in the reference
    assert len(np.unique(expected)) > 10
    assert np.allclose(got, expected, rtol=1e-12, atol=0)
