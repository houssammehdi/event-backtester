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
    MarketView,
    Order,
    OrderError,
    OrderStatus,
    OrderType,
    RiskLimits,
    Strategy,
    StrategyContext,
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


class RandomOrders(Strategy):
    """Random orders of every type, brackets, OCO pairs and cancels."""

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        rng = self.rng
        for symbol in view.symbols:
            price, u = view.price(symbol), rng.random()
            qty = float(rng.integers(1, 400)) * (1 if rng.random() < 0.5 else -1)
            off = float(rng.uniform(0.003, 0.03))
            near = price * (1 - off) if qty > 0 else price * (1 + off)
            far = price * (1 + off) if qty > 0 else price * (1 - off)
            tif = GTC if rng.random() < 0.5 else TimeInForce.DAY
            if u < 0.1:
                ctx.order(symbol, qty, OrderType.LIMIT, limit_price=near, tif=tif)
            elif u < 0.2:
                ctx.order(symbol, qty, OrderType.STOP, stop_price=far, tif=tif)
            elif u < 0.25:
                ctx.order(
                    symbol, qty, OrderType.STOP_LIMIT, stop_price=far, limit_price=far, tif=tif
                )
            elif u < 0.3:
                ctx.order(symbol, qty, OrderType.TRAILING_STOP, trail_percent=off, tif=tif)
            elif u < 0.35:
                kind = OrderType.MARKET_ON_CLOSE if rng.random() < 0.5 else OrderType.MARKET
                ctx.order(symbol, qty, kind)
            elif u < 0.45:  # limit entry, stop-loss 2 x off away, target 3 x off away
                side = 1 if qty > 0 else -1
                stop, target = price * (1 - 2 * side * off), price * (1 + 3 * side * off)
                entry = OrderType.LIMIT
                ctx.bracket(
                    symbol, qty, entry, limit_price=near, stop_loss=stop, take_profit=target
                )
            elif u < 0.5:
                ctx.bracket(symbol, qty, trail_amount=price * off)
            elif u < 0.55:
                group = ctx.oco_group()
                ctx.order(symbol, qty, OrderType.LIMIT, limit_price=near, oco_group=group, tif=tif)
                ctx.order(symbol, qty, OrderType.STOP, stop_price=far, oco_group=group, tif=tif)
            elif u < 0.6 and ctx.open_orders(symbol):
                ctx.cancel(ctx.open_orders(symbol)[0].id)


@pytest.mark.parametrize(
    "config",
    [
        BacktestConfig(slippage=FixedBpsSlippage(3.0), commission=BpsCommission(1.0)),
        BacktestConfig(intrabar="best", strict_limits=True, max_participation=0.0001),
        BacktestConfig(intrabar="low_first", limits=RiskLimits(allow_short=False)),
    ],
    ids=["worst", "best-strict-capacity", "low-first-long-only"],
)
def test_random_order_flow_keeps_every_book_consistent(config: BacktestConfig) -> None:
    """The engine checks both accounting identities on every bar; here the order
    records, the fills and the positions must also agree with each other."""
    feed = DataFeed(generate_ohlcv(n_symbols=3, years=1, seed=9))
    result = config.run(feed, RandomOrders(4), check_invariants=True)
    fills, orders = result.fills, {o.id: o for o in result.orders}
    assert len(fills) > 100
    filled = fills.groupby("order_id")["quantity"].sum()
    for order in orders.values():
        assert filled.get(order.id, 0.0) == pytest.approx(order.filled_quantity)
        if order.status is OrderStatus.FILLED:
            assert order.filled_quantity == pytest.approx(order.quantity)
        assert (order.closed_at is not None) == order.status.is_terminal
    signed = fills["quantity"].where(fills["side"] == "buy", -fills["quantity"])
    position = signed.groupby(fills["symbol"]).sum()
    for symbol in feed.symbols:
        assert result.positions[symbol].iloc[-1] == pytest.approx(position.get(symbol, 0.0))
    for entry_id in {o.parent_id for o in orders.values() if o.parent_id is not None}:
        exits = [o for o in orders.values() if o.parent_id == entry_id]
        closed = sum(o.filled_quantity for o in exits)
        assert closed <= orders[entry_id].filled_quantity + 1e-9  # exits never overshoot
    groups: dict[int, list[Order]] = {}
    for order in orders.values():
        if order.oco_group is not None:
            groups.setdefault(order.oco_group, []).append(order)
    assert groups
    for members in groups.values():
        assert sum(o.filled_quantity > 0 for o in members) <= 1  # one cancels the other
