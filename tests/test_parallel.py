"""Parallel grid search and walk-forward give exactly the serial results."""

from __future__ import annotations

import functools
import pickle

import numpy as np
import pandas as pd
import pytest

from backtester import BpsCommission, ConfigError, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.research import grid_search, walk_forward, walk_forward_windows
from backtester.research.parallel import available_cpus, resolve_jobs
from backtester.strategies import SmaCrossover

CONFIG = BacktestConfig(
    slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), max_participation=0.1
)
GRID = {"fast": [10, 20], "slow": [60, 120]}


@pytest.fixture(scope="module")
def feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=3, years=4, seed=5, missing_prob=0.01))


def test_parallel_grid_search_is_the_serial_one(feed: DataFeed) -> None:
    serial = grid_search(feed, SmaCrossover, GRID, config=CONFIG, start=150)
    parallel = grid_search(feed, SmaCrossover, GRID, config=CONFIG, start=150, n_jobs=2)
    pd.testing.assert_frame_equal(parallel.table, serial.table, check_exact=True)
    pd.testing.assert_frame_equal(parallel.returns, serial.returns, check_exact=True)
    assert parallel.best_params == serial.best_params
    assert parallel.deflated_sharpe == serial.deflated_sharpe
    assert parallel.lookback == serial.lookback == 120
    pd.testing.assert_series_equal(parallel.best_result.equity, serial.best_result.equity)
    pd.testing.assert_frame_equal(parallel.best_result.fills, serial.best_result.fills)
    # repr: unfilled orders hold NaN fields, and NaN != NaN once unpickled
    assert list(map(repr, parallel.best_result.orders)) == list(
        map(repr, serial.best_result.orders)
    )


def test_parallel_walk_forward_is_the_serial_one(feed: DataFeed) -> None:
    windows = walk_forward_windows(len(feed), 300, 200, start=120)
    factory = functools.partial(SmaCrossover, rebalance="daily")  # picklable
    serial = walk_forward(feed, factory, GRID, windows, config=CONFIG)
    parallel = walk_forward(feed, factory, GRID, windows, config=CONFIG, n_jobs=2)
    pd.testing.assert_frame_equal(parallel.windows, serial.windows, check_exact=True)
    pd.testing.assert_series_equal(parallel.oos_returns, serial.oos_returns, check_exact=True)
    pd.testing.assert_series_equal(parallel.oos_equity, serial.oos_equity, check_exact=True)
    assert parallel.n_trials == serial.n_trials == 4


def test_n_jobs_is_validated_and_lambdas_are_refused(feed: DataFeed) -> None:
    assert resolve_jobs(-1) == available_cpus() >= 1
    assert resolve_jobs(3) == 3
    for bad in (0, -2):
        with pytest.raises(ConfigError, match="n_jobs"):
            grid_search(feed, SmaCrossover, GRID, n_jobs=bad)
    with pytest.raises(ConfigError, match="picklable strategy factory"):
        grid_search(feed, lambda **kw: SmaCrossover(**kw), GRID, n_jobs=2)


def test_a_pickled_feed_stays_read_only(feed: DataFeed) -> None:
    copy = pickle.loads(pickle.dumps(feed))
    assert copy.symbols == feed.symbols
    assert copy.index.equals(feed.index)
    for field in ("open", "close", "volume"):
        np.testing.assert_array_equal(copy.array(field), feed.array(field))
        assert not copy.array(field).flags.writeable
    assert not copy.has_bar.flags.writeable
    assert not copy.last_close.flags.writeable
    assert copy.view(30).timestamp == feed.view(30).timestamp
    with pytest.raises(ValueError, match="read-only"):
        copy.view(30).window("close")[0, 0] = 1.0
