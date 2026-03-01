from __future__ import annotations

import copy
import math
from dataclasses import fields
from typing import Any

import pandas as pd
import pytest

from backtester import (
    CancelEvent,
    EventQueue,
    FillEvent,
    MarketEvent,
    Order,
    OrderError,
    OrderEvent,
    OrderStatus,
    OrderType,
    Side,
    SignalEvent,
    TargetEvent,
    TimeInForce,
)
from backtester.orders import validate_order_spec

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


M, L, S, SL, TS = (
    OrderType.MARKET,
    OrderType.LIMIT,
    OrderType.STOP,
    OrderType.STOP_LIMIT,
    OrderType.TRAILING_STOP,
)


@pytest.mark.parametrize(
    ("quantity", "kind", "prices", "message"),
    [
        (0.0, M, {}, "order quantity must be a positive finite number, got 0.0"),
        (math.nan, M, {}, "order quantity must be a positive finite number, got nan"),
        (1.0, M, {"limit_price": 9.0}, "market orders must not set limit_price"),
        (1.0, OrderType.MARKET_ON_OPEN, {"stop_price": 9.0}, "must not set stop_price"),
        (1.0, OrderType.MARKET_ON_CLOSE, {"trail_percent": 0.1}, "must not set a trail"),
        (1.0, L, {}, "limit orders require limit_price"),
        (1.0, L, {"limit_price": -1.0}, "limit_price must be a positive finite number, got -1.0"),
        (1.0, L, {"limit_price": 9.0, "stop_price": 9.0}, "limit orders must not set stop_price"),
        (1.0, S, {"stop_price": math.inf}, "stop_price must be a positive finite number"),
        (1.0, SL, {"limit_price": 9.0}, "stop_limit orders require stop_price"),
        (1.0, TS, {}, "trailing stops need exactly one of trail_amount and trail_percent"),
        (1.0, TS, {"trail_amount": 1.0, "trail_percent": 0.1}, "exactly one"),
        (1.0, TS, {"trail_percent": 1.5}, "trail_percent must be in (0, 1), got 1.5"),
        (1.0, TS, {"trail_amount": -2.0}, "trail_amount must be positive, got -2.0"),
        (1.0, L, {"limit_price": 9.0, "trail_amount": 1.0}, "limit orders must not set a trail"),
    ],
)
def test_order_spec_validation_messages(
    quantity: float, kind: OrderType, prices: dict[str, float], message: str
) -> None:
    limit, stop = prices.pop("limit_price", None), prices.pop("stop_price", None)
    with pytest.raises(OrderError) as excinfo:
        validate_order_spec(quantity, kind, limit, stop, **prices)
    assert message in str(excinfo.value)


@pytest.mark.parametrize(
    ("kind", "prices"),
    [
        (M, {}),
        (OrderType.MARKET_ON_OPEN, {}),
        (OrderType.MARKET_ON_CLOSE, {}),
        (L, {"limit_price": 9.0}),
        (S, {"stop_price": 9.0}),
        (SL, {"limit_price": 9.0, "stop_price": 9.5}),
        (TS, {"trail_percent": 0.05}),
        (TS, {"trail_amount": 0.5}),
    ],
)
def test_valid_order_specs(kind: OrderType, prices: dict[str, float]) -> None:
    limit, stop = prices.pop("limit_price", None), prices.pop("stop_price", None)
    validate_order_spec(2.0, kind, limit, stop, **prices)


def test_order_copy_is_complete_and_independent() -> None:
    values: dict[str, Any] = {
        "id": 7,
        "symbol": "A",
        "side": Side.SELL,
        "quantity": 30.0,
        "order_type": OrderType.TRAILING_STOP,
        "created_at": T0,
        "tif": TimeInForce.GTC,
        "limit_price": None,
        "stop_price": None,
        "tag": "stop_loss",
        "filled_quantity": 10.0,
        "avg_fill_price": 99.5,
        "status": OrderStatus.PARTIALLY_FILLED,
        "triggered": True,
        "closed_at": T1,
        "trail_amount": None,
        "trail_percent": 0.05,
        "trail_reference": 104.0,
        "parent_id": 6,
        "oco_group": 3,
        "armed": False,
    }
    assert set(values) == {f.name for f in fields(Order)}  # extend the test with the class
    order = Order(**values)
    clone = copy.copy(order)
    assert clone is not order
    assert clone == order
    assert {name: getattr(clone, name) for name in values} == values
    clone.record_fill(20.0, 98.0)
    assert (order.filled_quantity, order.status) == (10.0, OrderStatus.PARTIALLY_FILLED)


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
