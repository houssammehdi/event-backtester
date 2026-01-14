from __future__ import annotations

import pandas as pd
import pytest

from backtester import (
    CancelEvent,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderError,
    OrderEvent,
    OrderType,
    Side,
    SignalEvent,
    TargetEvent,
)

T0 = pd.Timestamp("2024-01-02")
T1 = pd.Timestamp("2024-01-03")


def fill(ts: pd.Timestamp, order_id: int = 1) -> FillEvent:
    return FillEvent(ts, order_id, "A", Side.BUY, 1.0, 10.0, 0.0)


def test_queue_orders_by_timestamp_then_priority_then_fifo() -> None:
    q = EventQueue()
    order = OrderEvent(T0, order_id=9, symbol="A", side=Side.BUY, quantity=1)
    target = TargetEvent(T0, weights={"A": 1.0})
    later_market = MarketEvent(T1, index=1)
    market = MarketEvent(T0, index=0)
    q.push(later_market)
    q.push(order)
    q.push(target)
    q.push(market)
    q.push(fill(T0, 1))
    q.push(fill(T0, 2))
    popped = [q.pop() for _ in range(len(q))]
    assert [type(e).__name__ for e in popped] == [
        "FillEvent",
        "FillEvent",
        "MarketEvent",
        "TargetEvent",
        "OrderEvent",
        "MarketEvent",
    ]
    fills = [e for e in popped if isinstance(e, FillEvent)]
    assert [f.order_id for f in fills] == [1, 2]  # FIFO among equal keys
    assert popped[-1] is later_market


def test_queue_empty_behaviour() -> None:
    q = EventQueue()
    assert not q
    assert q.peek() is None
    with pytest.raises(IndexError):
        q.pop()
    q.push(MarketEvent(T0, index=0))
    assert q.peek() is not None
    assert len(q) == 1


def test_target_weights_are_immutable() -> None:
    source = {"A": 0.5}
    ev = TargetEvent(T0, weights=source)
    source["A"] = 1.0  # the event keeps its own copy
    assert ev.weights["A"] == 0.5
    with pytest.raises(TypeError):
        ev.weights["A"] = 2.0  # type: ignore[index]


def test_events_validate_order_specs() -> None:
    with pytest.raises(OrderError):
        SignalEvent(T0, order_id=1, symbol="A", quantity=5, order_type=OrderType.LIMIT)
    with pytest.raises(OrderError):
        SignalEvent(T0, order_id=1, symbol="A", quantity=0)
    with pytest.raises(OrderError):
        OrderEvent(T0, order_id=1, symbol="A", side=Side.BUY, quantity=1, limit_price=10.0)
    with pytest.raises(OrderError):
        OrderEvent(
            T0,
            order_id=1,
            symbol="A",
            side=Side.BUY,
            quantity=1,
            order_type=OrderType.STOP,
            stop_price=-1.0,
        )


def test_fill_event_helpers() -> None:
    f = FillEvent(T0, 1, "A", Side.SELL, 3.0, 10.0, 0.5)
    assert f.signed_quantity == -3.0
    assert f.notional == 30.0
    assert CancelEvent(T0).order_id is None


def test_side_from_quantity() -> None:
    assert Side.from_quantity(2) is Side.BUY
    assert Side.from_quantity(-2) is Side.SELL
    with pytest.raises(OrderError):
        Side.from_quantity(0)
