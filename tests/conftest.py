import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_data():
    """Build a long format DataFrame from a mapping of station id to values."""
    def _make(
            series: dict[int, list[float]],
            freq: str = '5min',
            start: str = '2024-05-01 00:05',
    ) -> pd.DataFrame:
        length = {len(v) for v in series.values()}
        assert len(length) == 1, 'all stations need the same number of intervals'
        times = pd.date_range(start, periods=length.pop(), freq=freq, tz='UTC')
        wide = pd.DataFrame(series, index=times, dtype=np.float64)
        long = wide.melt(
            ignore_index=False,
            var_name='intern_id',
            value_name='precip',
        )
        return long.rename_axis('date').reset_index()[
            ['intern_id', 'date', 'precip']
        ]
    return _make


@pytest.fixture
def flags_of():
    """Get the flags of a single station in chronological order."""
    def _flags(data: pd.DataFrame, station_id: int, column: str) -> list[float]:
        rows = data[data['intern_id'] == station_id].sort_values('date')
        return rows[column].tolist()
    return _flags
