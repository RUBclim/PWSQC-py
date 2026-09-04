from datetime import datetime
from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd
import pytest

BACKENDS = ['pandas']
try:  # pragma: no cover
    import polars  # noqa: F401
except ImportError:  # pragma: no cover
    pass
else:  # pragma: no cover
    BACKENDS.append('polars')


@pytest.fixture(params=BACKENDS)
def backend(request):
    """The DataFrame library the test runs against."""
    return request.param


@pytest.fixture
def dates(backend):
    """Build a timezone aware timestamp column of the backend under test."""
    def _dates(*stamps: str) -> Any:
        stamps_ = [
            datetime.fromisoformat(s).replace(tzinfo=UTC) for s in stamps
        ]
        if backend == 'polars':
            import polars as pl

            return pl.Series('date', stamps_, dtype=pl.Datetime('us', 'UTC'))
        return pd.to_datetime(stamps_, utc=True)
    return _dates


@pytest.fixture
def frame(backend):
    """Build a DataFrame of the backend under test from a mapping of columns."""
    def _frame(columns: dict[str, Any]) -> Any:
        if backend == 'polars':
            import polars as pl

            return pl.DataFrame({
                name: (
                    pl.Series(name, values, dtype=pl.Float64)
                    if _is_float_list(values) else values
                )
                for name, values in columns.items()
            })
        return pd.DataFrame(columns)
    return _frame


def _is_float_list(values: Any) -> bool:
    return (
        isinstance(values, list) and
        bool(values) and
        all(isinstance(v, float) for v in values)
    )


@pytest.fixture
def make_data(backend):
    """Build a long format DataFrame from a mapping of station id to values."""
    def _make(
            series: dict[int, list[float]],
            freq: str = '5min',
            start: str = '2024-05-01 00:05',
    ) -> Any:
        length = {len(v) for v in series.values()}
        assert len(length) == 1, 'all stations need the same number of intervals'
        n = length.pop()
        times = pd.date_range(start, periods=n, freq=freq, tz='UTC')
        long = pd.DataFrame({
            'intern_id': np.repeat(list(series), n),
            'date': np.tile(times, len(series)),
            'precip': np.concatenate(
                [np.asarray(v, dtype=np.float64) for v in series.values()],
            ),
        })
        long['date'] = long['date'].dt.tz_convert('UTC')
        if backend == 'polars':
            import polars as pl

            return pl.from_pandas(long)
        return long
    return _make


def _to_list(data: Any, column: str) -> list[Any]:
    """The values of a column, a missing value always spelled ``None``."""
    if type(data).__module__.split('.')[0] == 'polars':
        return [None if v is not None and v != v else v for v in data[column]]
    return [None if v != v else v for v in data[column].tolist()]


@pytest.fixture
def values_of():
    """Get the values of a column, missing values become ``None``."""
    return _to_list


@pytest.fixture
def flags_of():
    """Get the flags of a single station in chronological order."""
    def _flags(data: Any, station_id: int, column: str) -> list[Any]:
        if type(data).__module__.split('.')[0] == 'polars':
            import polars as pl

            rows = data.filter(pl.col('intern_id') == station_id).sort('date')
        else:
            rows = data[data['intern_id'] == station_id].sort_values('date')
        return _to_list(rows, column)
    return _flags


@pytest.fixture
def select(backend):
    """Filter the rows of a DataFrame of either backend by station id."""
    def _select(data: Any, station_id: int) -> Any:
        if backend == 'polars':
            import polars as pl

            return data.filter(pl.col('intern_id') == station_id)
        return data[data['intern_id'] == station_id]
    return _select


@pytest.fixture
def set_flag(backend):
    """Overwrite a flag column for one station."""
    def _set(data: Any, station_id: int, column: str, value: int) -> Any:
        if backend == 'polars':
            import polars as pl

            return data.with_columns(
                pl.when(pl.col('intern_id') == station_id)
                .then(value)
                .otherwise(pl.col(column))
                .cast(pl.Int8)
                .alias(column),
            )
        result = data.copy()
        result.loc[result['intern_id'] == station_id, column] = value
        return result
    return _set


@pytest.fixture
def concat(backend):
    """Stack DataFrames of the backend under test on top of each other."""
    def _concat(*frames: Any) -> Any:
        if backend == 'polars':
            import polars as pl

            return pl.concat(frames)
        return pd.concat(frames)
    return _concat


@pytest.fixture
def drop_row(backend):
    """Drop one row by its position."""
    def _drop(data: Any, position: int) -> Any:
        keep = [i for i in range(len(data)) if i != position]
        if backend == 'polars':
            return data[keep]
        return data.iloc[keep]
    return _drop
