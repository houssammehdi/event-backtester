from __future__ import annotations

from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from backtester import DataFeed, MarketView, Strategy, StrategyContext, generate_ohlcv

Row = tuple[float, float, float, float, float]


def bars(rows: Sequence[Row], start: str = "2024-01-01") -> pd.DataFrame:
    """Single-symbol OHLCV frame on business days from ``(o, h, l, c, v)`` tuples."""
    index = pd.bdate_range(start, periods=len(rows), name="date")
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def flat_bars(n: int, price: float = 100.0, volume: float = 1e6) -> pd.DataFrame:
    return bars([(price, price, price, price, volume)] * n)


class Scripted(Strategy):
    """Runs ``actions[position](view, ctx)`` on the given bars; records every view seen."""

    name = "scripted"

    def __init__(self, actions: dict[int, Callable[[MarketView, StrategyContext], None]]) -> None:
        self.actions = actions
        self.seen: list[pd.Timestamp] = []

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        self.seen.append(view.timestamp)
        action = self.actions.get(view.position)
        if action is not None:
            action(view, ctx)


@pytest.fixture(scope="session")
def synthetic_feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=4, years=3, seed=7))


@pytest.fixture(scope="session")
def gappy_feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=4, years=3, seed=11, missing_prob=0.03))
