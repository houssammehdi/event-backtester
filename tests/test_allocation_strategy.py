"""AllocationStrategy and the stateless shortcut of TargetWeightStrategy."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from backtester import BpsCommission, ConfigError, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.portfolio import ALLOCATORS, MinimumVariance, SampleCovariance
from backtester.research import run_vectorized
from backtester.strategies import AllocationStrategy
from backtester.strategy.rebalancing import FloatArray

CFG = BacktestConfig(slippage=FixedBpsSlippage(3), commission=BpsCommission(1), lot_size=None)


@pytest.fixture(scope="module")
def feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=5, years=3, seed=13))


class Counting(AllocationStrategy):
    calls: ClassVar[list[int]] = []

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        Counting.calls.append(len(closes))
        return super().desired_weights(closes)


class EveryBar(AllocationStrategy):
    stateless: ClassVar[bool] = False


@pytest.mark.parametrize("method", sorted(ALLOCATORS))
def test_event_and_vectorized_paths_agree_for_every_method(method: str, feed: DataFeed) -> None:
    event = CFG.run(feed, AllocationStrategy(method, lookback=126))
    fast = run_vectorized(
        feed, AllocationStrategy(method, lookback=126), slippage_bps=3, commission_bps=1
    )
    np.testing.assert_allclose(fast.equity.to_numpy(), event.equity.to_numpy(), rtol=1e-9)
    assert len(event.fills) > 0


def test_stateless_strategy_only_evaluates_candidate_bars(feed: DataFeed) -> None:
    Counting.calls.clear()
    stateless = CFG.run(feed, Counting("erc", lookback=126))
    n_calls = len(Counting.calls)
    months = pd.DatetimeIndex(feed.index).to_period("M").nunique()
    # every bar until the first full window, then about one bar per month
    assert n_calls <= 127 + months
    every_bar = CFG.run(feed, EveryBar("erc", lookback=126))
    pd.testing.assert_series_equal(stateless.equity, every_bar.equity)
    pd.testing.assert_frame_equal(stateless.fills, every_bar.fills)


def test_stateless_warm_up_establishes_the_last_scheduled_portfolio(feed: DataFeed) -> None:
    start = 400  # not the first bar of a month
    assert feed.index[start].month == feed.index[start - 1].month
    a = CFG.run(feed, AllocationStrategy("iv", lookback=126), start=start)
    b = CFG.run(feed, EveryBar("iv", lookback=126), start=start)
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert a.fills["timestamp"].min() == feed.index[start + 1]


def test_unlisted_symbols_get_zero_weight_until_they_have_a_window() -> None:
    frames = generate_ohlcv(n_symbols=3, years=2, seed=3)
    frames["SYN03"] = frames["SYN03"].iloc[300:]  # lists 300 bars later
    feed = DataFeed(frames)
    strategy = AllocationStrategy("iv", lookback=100)
    closes = feed.last_close
    early = strategy.desired_weights(closes[200:301])
    assert early is not None
    assert early[2] == 0.0
    assert early.sum() == pytest.approx(1.0)
    late = strategy.desired_weights(closes[400:501])
    assert late is not None
    assert late[2] > 0


def test_allocator_instances_and_validation() -> None:
    custom = AllocationStrategy(MinimumVariance(covariance=SampleCovariance(), max_weight=0.5))
    assert custom.params()["method"] == "minvar"
    capped = AllocationStrategy("minvar", max_weight=0.4)
    assert capped._allocator == MinimumVariance(max_weight=0.4)
    with pytest.raises(ConfigError, match="unknown allocation method"):
        AllocationStrategy("magic")
    with pytest.raises(ConfigError, match="covariance"):
        AllocationStrategy("iv", covariance="shrunk")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="lookback"):
        AllocationStrategy("iv", lookback=1)
