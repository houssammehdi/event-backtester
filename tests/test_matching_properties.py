"""Property tests of the intrabar path simulation against independent references."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester import OrderEvent, OrderStatus, OrderType, Side, SimulatedBroker, TimeInForce
from backtester.data import Bar, DataFeed
from backtester.execution import NoCommission, NoSlippage
from backtester.execution.matching import BarSimulation, idle_on_bar, path_points
from backtester.orders import Order

T0 = pd.Timestamp("2024-01-01")
T1 = pd.Timestamp("2024-01-02")


@st.composite
def ohlc(draw: st.DrawFn) -> Bar:
    o = draw(st.floats(90, 110))
    c = draw(st.floats(90, 110))
    h = max(o, c) + draw(st.floats(0, 6))
    lo = min(o, c) - draw(st.floats(0, 6))
    return Bar(T1, "A", round(o, 2), round(h, 2), round(lo, 2), round(c, 2), 1e6)


def simulate(
    orders: list[Order],
    bar: Bar,
    high_first: bool,
    children: dict[int, list[int]] | None = None,
) -> BarSimulation:
    by_id = {o.id: o for o in orders}
    sim = BarSimulation(
        orders,
        by_id.__getitem__,
        children or {},
        bar,
        high_first=high_first,
        capacity=math.inf,
        slippage=NoSlippage(),
        commission=NoCommission(),
        strict_limits=False,
    )
    sim.run()
    return sim


def old_rule(order: Order, bar: Bar) -> float | None:
    """The per-order matching rule of version 0.1 (no path, gaps at the open)."""
    buy = order.side is Side.BUY
    if order.order_type is OrderType.LIMIT:
        assert order.limit_price is not None
        limit = order.limit_price
        if (buy and bar.open <= limit) or (not buy and bar.open >= limit):
            return bar.open
        reached = bar.low <= limit if buy else bar.high >= limit
        return limit if reached else None
    assert order.stop_price is not None
    stop = order.stop_price
    if (buy and bar.open >= stop) or (not buy and bar.open <= stop):
        return bar.open
    triggered = bar.high >= stop if buy else bar.low <= stop
    return stop if triggered else None


@settings(max_examples=300, deadline=None)
@given(
    bar=ohlc(),
    specs=st.lists(
        st.tuples(
            st.sampled_from([OrderType.LIMIT, OrderType.STOP]),
            st.sampled_from([Side.BUY, Side.SELL]),
            st.floats(85, 115),
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_independent_orders_ignore_the_path_and_match_the_old_rule(
    bar: Bar, specs: list[tuple[OrderType, Side, float]]
) -> None:
    def make() -> list[Order]:
        out = []
        for k, (kind, side, price) in enumerate(specs, start=1):
            key = "limit_price" if kind is OrderType.LIMIT else "stop_price"
            prices = {key: round(price, 2)}
            out.append(Order(k, "A", side, 1.0, kind, T0, TimeInForce.GTC, **prices))
        return out

    a, b = make(), make()
    fills_a = {f.order_id: f.price for f in simulate(a, bar, True).fills}
    fills_b = {f.order_id: f.price for f in simulate(b, bar, False).fills}
    assert fills_a == fills_b
    for order in make():
        expected = old_rule(order, bar)
        assert fills_a.get(order.id) == expected


def dense_trailing(
    order_side: Side, pct: float, reference: float, bar: Bar, high_first: bool
) -> float | None:
    """Walk the path in small steps: follow the best price, exit when the level is crossed."""
    points = path_points(bar, high_first)
    ref = reference
    level = lambda r: r * (1 - pct) if order_side is Side.SELL else r * (1 + pct)  # noqa: E731
    first = points[0]
    ref = max(ref, first) if order_side is Side.SELL else min(ref, first)
    if (order_side is Side.SELL and first <= level(ref)) or (
        order_side is Side.BUY and first >= level(ref)
    ):
        return first
    for a, b in itertools.pairwise(points):
        steps = max(int(abs(b - a) / 0.01), 1)  # crossings are resolved exactly
        previous = a
        for p in np.linspace(a, b, steps + 1)[1:]:
            stop = level(ref)
            crossed = previous > stop >= p if order_side is Side.SELL else previous < stop <= p
            if crossed:
                return stop
            ref = max(ref, p) if order_side is Side.SELL else min(ref, p)
            previous = p
    return None


@settings(max_examples=200, deadline=None)
@given(
    bar=ohlc(),
    side=st.sampled_from([Side.BUY, Side.SELL]),
    pct=st.floats(0.005, 0.08),
    reference=st.floats(90, 110),
    high_first=st.booleans(),
)
def test_trailing_stop_matches_a_dense_path_walk(
    bar: Bar, side: Side, pct: float, reference: float, high_first: bool
) -> None:
    order = Order(
        1,
        "A",
        side,
        1.0,
        OrderType.TRAILING_STOP,
        T0,
        TimeInForce.GTC,
        trail_percent=round(pct, 4),
        trail_reference=round(reference, 2),
    )
    expected = dense_trailing(side, round(pct, 4), round(reference, 2), bar, high_first)
    fills = simulate([order], bar, high_first).fills
    if expected is None:
        assert fills == []
    else:
        assert len(fills) == 1
        assert fills[0].price == pytest.approx(expected, rel=1e-9, abs=1e-9)


@settings(max_examples=150, deadline=None)
@given(bar=ohlc(), stop_gap=st.floats(0.5, 8), target_gap=st.floats(0.5, 8))
def test_worst_case_is_never_better_than_either_path(
    bar: Bar, stop_gap: float, target_gap: float
) -> None:
    rows = [(100, 100, 100, 100, 1e6), (bar.open, bar.high, bar.low, bar.close, 1e6)]
    frame = pd.DataFrame(
        rows,
        index=pd.bdate_range("2024-01-01", periods=2),
        columns=["open", "high", "low", "close", "volume"],
    )
    feed = DataFeed({"A": frame})
    values = {}
    for policy in ("worst", "best", "high_first", "low_first"):
        broker = SimulatedBroker(intrabar=policy)  # type: ignore[arg-type]
        ts = feed.index[0]
        broker.submit(OrderEvent(ts, order_id=1, symbol="A", side=Side.BUY, quantity=10))
        broker.submit(
            OrderEvent(
                ts,
                order_id=2,
                symbol="A",
                side=Side.SELL,
                quantity=10,
                order_type=OrderType.STOP,
                stop_price=bar.open - stop_gap,
                tif=TimeInForce.GTC,
                parent_id=1,
            )
        )
        broker.submit(
            OrderEvent(
                ts,
                order_id=3,
                symbol="A",
                side=Side.SELL,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=bar.open + target_gap,
                tif=TimeInForce.GTC,
                parent_id=1,
            )
        )
        fills = broker.process_bar(feed.view(1))
        values[policy] = sum(f.signed_quantity * (bar.close - f.price) for f in fills)
        exits = [broker.get(i).status for i in (2, 3)]
        assert exits.count(OrderStatus.FILLED) <= 1  # never both exits
    assert values["worst"] <= min(values["high_first"], values["low_first"]) + 1e-9
    assert values["best"] >= max(values["high_first"], values["low_first"]) - 1e-9


@settings(max_examples=200, deadline=None)
@given(
    bar=ohlc(),
    specs=st.lists(
        st.tuples(
            st.sampled_from([OrderType.LIMIT, OrderType.STOP]),
            st.sampled_from([Side.BUY, Side.SELL]),
            st.floats(0.01, 5),
            st.lists(st.sampled_from(list(OrderType)), max_size=2),
            st.booleans(),
        ),
        min_size=1,
        max_size=4,
    ),
)
def test_idle_orders_do_nothing_on_either_path(
    bar: Bar,
    specs: list[tuple[OrderType, Side, float, list[OrderType], bool]],
) -> None:
    """The broker skips a symbol whose orders are all idle; simulating it would be a no-op.

    Each entry is a limit or stop the bar does not reach, with unarmed exits of any type
    and possibly an OCO group.
    """

    def make() -> tuple[list[Order], dict[int, list[int]]]:
        orders: list[Order] = []
        children: dict[int, list[int]] = {}
        for kind, side, gap, exit_types, oco in specs:
            beyond_high = (kind is OrderType.LIMIT) is (side is Side.SELL)
            price = round(bar.high + gap if beyond_high else bar.low - gap, 2)
            key = "limit_price" if kind is OrderType.LIMIT else "stop_price"
            entry = Order(
                len(orders) + 1,
                "A",
                side,
                5.0,
                kind,
                T0,
                TimeInForce.GTC,
                oco_group=1 if oco else None,
                **{key: price},
            )
            orders.append(entry)
            for exit_type in exit_types:
                prices: dict[str, float] = {}
                if exit_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                    prices["limit_price"] = 100.0
                if exit_type in (OrderType.STOP, OrderType.STOP_LIMIT):
                    prices["stop_price"] = 100.0
                if exit_type is OrderType.TRAILING_STOP:
                    prices["trail_percent"] = 0.01
                exit_order = Order(
                    len(orders) + 1,
                    "A",
                    Side.SELL if side is Side.BUY else Side.BUY,
                    5.0,
                    exit_type,
                    T0,
                    TimeInForce.GTC,
                    parent_id=entry.id,
                    armed=False,
                    **prices,
                )
                orders.append(exit_order)
                children.setdefault(entry.id, []).append(exit_order.id)
        return orders, children

    reference, _ = make()
    assert all(idle_on_bar(o, bar) for o in reference)
    for high_first in (True, False):
        orders, children = make()
        assert simulate(orders, bar, high_first, children).fills == []
        assert orders == reference


def test_orders_the_bar_reaches_are_not_idle() -> None:
    bar = Bar(T1, "A", 100.0, 104.0, 97.0, 101.0, 1e6)

    def order(side: Side, kind: OrderType, price: float, **kw: object) -> Order:
        key = "limit_price" if kind is OrderType.LIMIT else "stop_price"
        return Order(1, "A", side, 1.0, kind, T0, **{key: price}, **kw)  # type: ignore[arg-type]

    assert not idle_on_bar(order(Side.BUY, OrderType.LIMIT, 97.0), bar)  # touches the low
    assert not idle_on_bar(order(Side.SELL, OrderType.STOP, 97.0), bar)
    assert not idle_on_bar(order(Side.BUY, OrderType.STOP, 104.0), bar)
    assert idle_on_bar(order(Side.BUY, OrderType.STOP, 104.01), bar)
    assert idle_on_bar(order(Side.BUY, OrderType.STOP, 90.0, armed=False), bar)
    assert not idle_on_bar(Order(2, "A", Side.BUY, 1.0, OrderType.MARKET, T0), bar)
