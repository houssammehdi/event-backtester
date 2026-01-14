from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from backtester import DataFeed, generate_ohlcv

Row = tuple[float, float, float, float, float]


def bars(rows: Sequence[Row], start: str = "2024-01-01") -> pd.DataFrame:
    """Single-symbol OHLCV frame on business days from ``(o, h, l, c, v)`` tuples."""
    index = pd.bdate_range(start, periods=len(rows), name="date")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def flat_bars(n: int, price: float = 100.0, volume: float = 1e6) -> pd.DataFrame:
    return bars([(price, price, price, price, volume)] * n)


@pytest.fixture(scope="session")
def synthetic_feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=4, years=3, seed=7))


@pytest.fixture(scope="session")
def gappy_feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=4, years=3, seed=11, missing_prob=0.03))
