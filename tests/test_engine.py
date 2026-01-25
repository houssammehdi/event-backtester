"""End-to-end behaviour of the event loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import (
    BpsCommission,
    ConfigError,
    DataFeed,
    Engine,
    FixedBpsSlippage,
    LookAheadError,
    MarketView,
    OrderError,
    OrderStatus,
    OrderType,
    Portfolio,
    RiskLimits,
    SimulatedBroker,
    SquareRootImpactSlippage,
    Strategy,
    StrategyContext,
    TimeInForce,
)
from backtester.config import BacktestConfig
from backtester.strategies import STRATEGIES
from tests.conftest import Scripted, bars


def ramp_feed(n: int = 6) -> DataFeed:
    rows = [(100 + i, 101 + i, 99 + i, 100.5 + i, 1e6) for i in range(n)]
    return DataFeed({"A": bars(rows)})


@pytest.mark.parametrize(
    "action",
    [
        lambda view, ctx: ctx.order("AA", 10),
        lambda view, ctx: ctx.target_weights({"AA": 1.0}),
        lambda view, ctx: ctx.target_weights({"A": 0.5, "AA": 0.5}, partial=True),
        lambda view, ctx: ctx.cancel(symbol="AA"),
    ],
)
def test_unknown_symbols_are_rejected_where_the_mistake_is_made(action) -> None:  # type: ignore[no-untyped-def]
    """Regression: a mistyped target was silently dropped, and a mistyped order raised a
    bare KeyError one bar later from inside the broker's matching loop."""
    with pytest.raises(OrderError, match="unknown symbol 'AA'"):
        Engine(ramp_feed(), Scripted({0: action})).run()


def test_context_without_a_universe_accepts_any_symbol() -> None:
    ctx = StrategyContext(Portfolio(1_000), SimulatedBroker())
    ctx._begin(pd.Timestamp("2024-01-02"))
    ctx.order("ANY", 1)
    ctx.target_weights({"ANY": 1.0})
    assert len(ctx._drain()) == 2


def test_market_order_fills_next_open_and_accounting_is_exact() -> None:
    feed = ramp_feed()
    strategy = Scripted({0: lambda v, ctx: ctx.order("A", 10)})
    result = Engine(feed, strategy, initial_cash=10_000, check_invariants=True).run()
    assert len(result.fills) == 1
    f = result.fills.iloc[0]
    assert f["timestamp"] == feed.index[1]
    assert f["price"] == 101  # open of bar 1
    assert result.equity.iloc[0] == 10_000  # nothing can fill on the decision bar
    assert result.equity.iloc[1] == pytest.approx(10_000 - 1010 + 10 * 101.5)
    assert result.positions["A"].tolist() == [0, 10, 10, 10, 10, 10]
    assert result.cash.iloc[-1] == pytest.approx(10_000 - 1010)


def test_limit_order_through_the_engine_and_context_views() -> None:
    feed = ramp_feed()
    seen: dict[str, object] = {}

    def place(view: MarketView, ctx: StrategyContext) -> None:
        seen["id"] = ctx.order(
            "A", 5, OrderType.LIMIT, limit_price=95.0, tif=TimeInForce.GTC, tag="dip"
        )

    def inspect(view: MarketView, ctx: StrategyContext) -> None:
        orders = ctx.open_orders("A")
        seen["open"] = [o.id for o in orders]
        orders[0].quantity = 999  # a copy: must not affect the broker
        ctx.cancel(seen["id"])  # type: ignore[arg-type]

    strategy = Scripted({0: place, 1: inspect})
    engine = Engine(feed, strategy)
    result = engine.run()
    assert seen["open"] == [seen["id"]]
    order = result.orders[0]
    assert order.quantity == 5
    assert order.status is OrderStatus.CANCELLED
    assert order.tag == "dip"
    assert order.filled_quantity == 0
    assert order.closed_at == feed.index[1]
    assert result.fills.empty


def test_strategy_cannot_read_the_future() -> None:
    class Peeker(Strategy):
        def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
            view.bar_at(view.timestamp + pd.Timedelta(days=1), "A")

    with pytest.raises(LookAheadError):
        Engine(ramp_feed(), Peeker()).run()


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_results_are_invariant_to_future_data(name: str, synthetic_feed: DataFeed) -> None:
    """Truncating the data after bar k must not change anything up to bar k."""
    cfg = BacktestConfig(slippage=FixedBpsSlippage(3), commission=BpsCommission(1))
    full = cfg.run(synthetic_feed, STRATEGIES[name]())
    k = 500
    cut = cfg.run(synthetic_feed.slice(0, k + 1), STRATEGIES[name]())
    pd.testing.assert_series_equal(full.equity.iloc[: k + 1], cut.equity)
    early = full.fills[full.fills["timestamp"] <= synthetic_feed.index[k]]
    pd.testing.assert_frame_equal(early.reset_index(drop=True), cut.fills)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_realistic_run_keeps_identities_with_missing_bars(name: str, gappy_feed: DataFeed) -> None:
    cfg = BacktestConfig(
        slippage=SquareRootImpactSlippage(eta=0.5, spread_bps=2),
        commission=BpsCommission(1),
        max_participation=0.05,
        limits=RiskLimits(max_gross_leverage=1.5, max_position_weight=0.6),
        borrow_rate=0.02,
    )
    result = cfg.run(gappy_feed, STRATEGIES[name](), check_invariants=True)
    values = (result.positions * result.prices).sum(axis=1)
    np.testing.assert_allclose(result.equity, result.cash + values, rtol=1e-10)
    assert (result.exposure["gross"] / result.equity).max() < 1.5 * 1.1  # drift allowance
    assert len(result.fills) > 0
    assert result.metrics().n_bars == len(gappy_feed)


def test_runs_are_deterministic(synthetic_feed: DataFeed) -> None:
    a = BacktestConfig().run(synthetic_feed, STRATEGIES["xsmom"]())
    b = BacktestConfig().run(synthetic_feed, STRATEGIES["xsmom"]())
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_trading_window_and_warmup(synthetic_feed: DataFeed) -> None:
    start, end = 300, 600
    result = BacktestConfig().run(synthetic_feed, STRATEGIES["tsmom"](), start=start, end=end)
    assert result.equity.index[0] == synthetic_feed.index[start]
    assert result.equity.index[-1] == synthetic_feed.index[end]
    assert result.equity.iloc[0] == result.initial_cash
    assert result.fills["timestamp"].min() == synthetic_feed.index[start + 1]
    # warm-up let the strategy trade immediately instead of waiting a year for history
    by_ts = BacktestConfig().run(
        synthetic_feed,
        STRATEGIES["tsmom"](),
        start=str(synthetic_feed.index[start].date()),
        end=synthetic_feed.index[end],
    )
    pd.testing.assert_series_equal(result.equity, by_ts.equity)


def test_engine_guards() -> None:
    feed = ramp_feed()
    engine = Engine(feed, Scripted({}))
    engine.run()
    with pytest.raises(RuntimeError):
        engine.run()
    with pytest.raises(ConfigError):
        Engine(feed, Scripted({}), start=4, end=2)
    with pytest.raises(ConfigError):
        Engine(feed, Scripted({}), start=99)
    with pytest.raises(ConfigError):
        Engine(feed, Scripted({}), periods_per_year=0)


def test_on_fill_hook_and_explicit_orders_with_risk() -> None:
    fills: list[float] = []

    class Hook(Scripted):
        def on_fill(self, fill, ctx) -> None:  # type: ignore[no-untyped-def]
            fills.append(fill.quantity)

    feed = ramp_feed()
    strategy = Hook({0: lambda v, ctx: ctx.order("A", 1_000)})
    risk = RiskLimits(max_position_weight=0.5)
    result = Engine(feed, strategy, initial_cash=10_000, risk=risk).run()
    assert fills == [49]  # 50% of 10k at the 100.5 close -> 49 shares
    assert any(a.kind == "clipped" for a in result.risk_log)


def test_report_mentions_key_sections(synthetic_feed: DataFeed) -> None:
    broker = SimulatedBroker(FixedBpsSlippage(2), BpsCommission(1), max_participation=0.1)
    result = Engine(synthetic_feed, STRATEGIES["sma"](), broker=broker).run()
    report = result.report()
    for text in ("Sharpe ratio", "Max drawdown", "Beta", "Hit rate", "Commission"):
        assert text in report
    flat = result.metrics().as_dict()
    assert "benchmark_beta" in flat
    assert "trade_hit_rate" in flat
