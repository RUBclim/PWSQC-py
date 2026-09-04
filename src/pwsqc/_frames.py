"""Minimal adapter over the DataFrame libraries the filters accept.

The heavy lifting of PWSQC happens in numpy, the DataFrame is only used to get
the columns in and the results back out. Everything the filters need of it is
collected here, so that both :mod:`pandas` and :mod:`polars` can be passed in
without the rest of the package knowing which one it is.
"""
from typing import Any
from typing import cast
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

#: The DataFrame a function was given, so that it is known to return one of the
#: same type. Neither library is imported here, both of them are optional.
FrameT = TypeVar('FrameT')


def is_polars(df: Any) -> bool:
    """Whether ``df`` is a :class:`polars.DataFrame`."""
    return type(df).__module__.split('.')[0] == 'polars'


def columns(df: Any) -> list[str]:
    """The column names of ``df``."""
    return list(df.columns)


def n_rows(df: Any) -> int:
    """The number of rows of ``df``."""
    return len(df)


def values(df: Any, column: str) -> NDArray[Any]:
    """The raw values of ``column`` as a numpy array."""
    if is_polars(df):
        return df[column].to_numpy()

    import pandas as pd

    series = df[column]
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        # a timezone aware column would come back as an object array of
        # Timestamps, the UTC values are what the reshaping needs
        series = series.dt.tz_convert('UTC').dt.tz_localize(None)
    return np.asarray(series)


def floats(df: Any, column: str) -> NDArray[np.float64]:
    """The values of ``column`` as float64, missing values become NaN."""
    if is_polars(df):
        import polars as pl

        array = df[column].cast(pl.Float64).to_numpy()
        return np.asarray(array, dtype=np.float64)
    return np.asarray(
        df[column].to_numpy(dtype=np.float64, na_value=np.nan),
        dtype=np.float64,
    )


def with_columns(df: FrameT, new: dict[str, NDArray[Any]]) -> FrameT:
    """Return ``df`` with the arrays added as columns, leaving the input untouched.

    A NaN of a floating point array becomes a null in polars, which is how a
    missing value is spelled there, and stays a NaN in pandas.
    """
    return cast('FrameT', _with_columns(df, new))


def _with_columns(df: Any, new: dict[str, NDArray[Any]]) -> Any:
    if is_polars(df):
        import polars as pl

        return df.with_columns([
            pl.Series(name, array).fill_nan(None)
            if array.dtype.kind == 'f' else pl.Series(name, array)
            for name, array in new.items()
        ])

    result = df.copy()
    for name, array in new.items():
        result[name] = array
    return result
