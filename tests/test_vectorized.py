"""The vectorized fast path must reproduce the event engine on the same strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import BpsCommission, ConfigError, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.research import run_vectorized
from backtester.strategies import STRATEGIES, SmaCrossover

SLIP, COMM = 4.0, 2.0
CFG = BacktestConfig(slippage=FixedBpsSlippage(SLIP), commission=BpsCommission(COMM), lot_size=None)


@pytest.fixture(scope="module")
def feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=5, years=4, seed=21))


@pytest.mark.parametrize("name", sorted(STRATEGIES))
@pytest.mark.parametrize("start", [None, 300])
def test_equity_parity(name: str, start: int | None, feed: DataFeed) -> None:
    event = CFG.run(feed, STRATEGIES[name](), start=start)
    fast = run_vectorized(
        feed, STRATEGIES[name](), slippage_bps=SLIP, commission_bps=COMM, start=start
    )
    np.testing.assert_allclose(fast.equity.to_numpy(), event.equity.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(fast.positions.to_numpy(), event.positions.to_numpy(), atol=1e-6)
    costs = event.total_commission + event.total_slippage
    assert fast.total_costs == pytest.approx(costs, rel=1e-9)
    assert fast.summary()["sharpe"] == pytest.approx(event.metrics().sharpe, rel=1e-6)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_bulk_weights_equal_incremental_weights(name: str, feed: DataFeed) -> None:
    """Each row of the bulk frame must be computable from the past alone."""
    bulk = STRATEGIES[name]().desired_weights_frame(
        pd.DataFrame(feed.last_close, index=feed.index, columns=feed.symbols)
    )
    incremental = STRATEGIES[name]()
    for i in range(len(feed)):
        w = incremental.desired_weights(feed.view(i).filled_closes(incremental.history_bars))
        row = bulk.iloc[i].to_numpy()
        if w is None:
            assert np.isnan(row).all()
        else:
            np.testing.assert_allclose(row, w, rtol=1e-9, atol=1e-12)


def test_parity_catches_a_leaky_vectorized_signal(feed: DataFeed) -> None:
    class Leaky(SmaCrossover):
        def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
            return super().desired_weights_frame(closes.shift(-1).ffill())  # peeks one bar

    event = CFG.run(feed, Leaky(fast=5, slow=20))
    fast = run_vectorized(feed, Leaky(fast=5, slow=20), slippage_bps=SLIP, commission_bps=COMM)
    assert not np.allclose(fast.equity.to_numpy(), event.equity.to_numpy(), rtol=1e-6)


def test_vectorized_rejects_missing_bars_and_bad_windows() -> None:
    gappy = DataFeed(generate_ohlcv(3, 1, seed=1, missing_prob=0.05))
    with pytest.raises(ConfigError, match="complete bars"):
        run_vectorized(gappy, SmaCrossover(fast=5, slow=20))
    full = DataFeed(generate_ohlcv(3, 1, seed=1))
    with pytest.raises(ConfigError):
        run_vectorized(full, SmaCrossover(fast=5, slow=20), start=100, end=50)
    with pytest.raises(ConfigError):
        run_vectorized(full, SmaCrossover(fast=5, slow=20), initial_cash=0)
