"""End-to-end behaviour of the event loop."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import (
    ConfigError,
    DataFeed,
    Engine,
    LookAheadError,
    MarketView,
    OrderStatus,
    OrderType,
    RiskLimits,
    Strategy,
    StrategyContext,
    TimeInForce,
)
from tests.conftest import Scripted, bars


def ramp_feed(n: int = 6) -> DataFeed:
    rows = [(100 + i, 101 + i, 99 + i, 100.5 + i, 1e6) for i in range(n)]
    return DataFeed({"A": bars(rows)})


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
