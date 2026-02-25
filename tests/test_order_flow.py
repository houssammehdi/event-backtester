"""Linked and trailing orders through the strategy context, the risk manager and the
engine, and market-on-close rebalancing against the vectorized close path."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from backtester import (
    BpsCommission,
    ConfigError,
    DataFeed,
    Engine,
    FixedBpsSlippage,
    OrderError,
    OrderStatus,
    OrderType,
    RiskLimits,
    TimeInForce,
    generate_ohlcv,
)
from backtester.config import BacktestConfig
from backtester.research import run_vectorized
from backtester.strategies import STRATEGIES
from tests.conftest import Scripted, bars

GTC = TimeInForce.GTC


def test_bracket_through_the_engine_and_risk() -> None:
    rows = [
        (100, 101, 99, 100, 1e6),
        (100, 101, 99, 100.5, 1e6),
        (100.5, 104, 100, 103, 1e6),
        (103, 106, 102, 105, 1e6),
        (105, 105.5, 97, 98, 1e6),
    ]
    feed = DataFeed({"A": bars(rows)})
    ids: dict[str, Any] = {}

    def enter(view: Any, ctx: Any) -> None:
        ids["long"] = ctx.bracket("A", 100, stop_loss=96.0, take_profit=105.0, tag="swing")

    result = Engine(feed, Scripted({0: enter}), initial_cash=100_000, check_invariants=True).run()
    b = ids["long"]
    orders = {o.id: o for o in result.orders}
    assert orders[b.stop_loss].tag == "swing:stop_loss"
    assert orders[b.take_profit].status is OrderStatus.FILLED  # 105 on bar 3
    assert orders[b.stop_loss].status is OrderStatus.CANCELLED
    trades = result.trades()
    assert len(trades) == 1
    assert trades.iloc[0]["exit_price"] == 105
    assert result.positions["A"].iloc[-1] == 0


def test_rejected_entry_rejects_its_exits_and_bracket_validation() -> None:
    feed = DataFeed({"A": bars([(100, 101, 99, 100, 1e6)] * 3)})
    ids: dict[str, Any] = {}

    def short(view: Any, ctx: Any) -> None:
        ids["b"] = ctx.bracket("A", -10, stop_loss=110.0, take_profit=90.0)

    result = Engine(feed, Scripted({0: short}), risk=RiskLimits(allow_short=False)).run()
    orders = {o.id: o for o in result.orders}
    b = ids["b"]
    assert [orders[i].status for i in (b.entry, b.stop_loss, b.take_profit)] == [
        OrderStatus.REJECTED
    ] * 3
    for kwargs, message in (
        ({"stop_loss": 110.0, "take_profit": 105.0}, "below"),
        ({}, "needs a stop_loss"),
        ({"stop_loss": 90.0, "trail_percent": 0.05}, "not both"),
    ):
        with pytest.raises(OrderError, match=message):
            Engine(feed, Scripted({0: lambda v, ctx, kw=kwargs: ctx.bracket("A", 10, **kw)})).run()


def test_oco_alternatives_are_not_counted_twice_by_the_position_limit() -> None:
    feed = DataFeed({"A": bars([(100, 101, 99, 100, 1e6)] * 3)})

    def straddle(view: Any, ctx: Any) -> None:
        group = ctx.oco_group()
        ctx.order("A", 400, OrderType.STOP, stop_price=110.0, oco_group=group)
        ctx.order("A", 400, OrderType.LIMIT, limit_price=90.0, oco_group=group)

    risk = RiskLimits(max_position_weight=0.5)  # 500 shares at 100 on 100k
    result = Engine(feed, Scripted({0: straddle}), initial_cash=100_000, risk=risk).run()
    assert [o.quantity for o in result.orders] == [400, 400]  # neither clipped


def test_trailing_stop_from_the_engine_starts_at_the_decision_close() -> None:
    """Decided at bar 1's close (100); bar 2 (101, 101.5, 96, 97) opens higher, which
    lifts the reference to 101 (level 97.97). Path A would ratchet to 101.5 first and
    sell at 98.455; the worst case, path B, sells at 97.97 on the way down."""
    rows = [(100, 100, 100, 100, 1e6), (100, 100.5, 99.5, 100, 1e6), (101, 101.5, 96, 97, 1e6)]
    feed = DataFeed({"A": bars(rows)})
    strategy = Scripted(
        {
            0: lambda v, ctx: ctx.order("A", 10),
            1: lambda v, ctx: ctx.order(
                "A", -10, OrderType.TRAILING_STOP, trail_percent=0.03, tif=GTC
            ),
        }
    )
    result = Engine(feed, strategy).run()
    stop = result.orders[1]
    assert stop.status is OrderStatus.FILLED
    assert stop.avg_fill_price == pytest.approx(101 * 0.97)


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_market_on_close_rebalancing_matches_the_vectorized_close_path(name: str) -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=4, years=3, seed=17))
    cfg = BacktestConfig(
        slippage=FixedBpsSlippage(3),
        commission=BpsCommission(1),
        lot_size=None,
        target_order_type=OrderType.MARKET_ON_CLOSE,
    )
    event = cfg.run(feed, STRATEGIES[name]())
    fast = run_vectorized(
        feed, STRATEGIES[name](), slippage_bps=3, commission_bps=1, execution="close"
    )
    np.testing.assert_allclose(fast.equity.to_numpy(), event.equity.to_numpy(), rtol=1e-9)
    assert event.orders[0].order_type is OrderType.MARKET_ON_CLOSE
    with pytest.raises(ConfigError, match="execution"):
        run_vectorized(feed, STRATEGIES[name](), execution="vwap")  # type: ignore[arg-type]
