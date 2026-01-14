"""Event types and the timestamp-ordered event queue that drives the engine.

Every event carries a timestamp and a class-level ``priority``. The queue orders events
by ``(timestamp, priority, insertion sequence)``, which fixes the causal order of work
inside a single bar:

1. :class:`FillEvent` (priority 0) - executions produced when the new bar printed.
2. :class:`MarketEvent` (priority 1) - the bar closed; mark-to-market, risk, strategy.
3. :class:`SignalEvent` / :class:`TargetEvent` / :class:`CancelEvent` (priority 2) -
   strategy intents, reviewed by the risk manager.
4. :class:`OrderEvent` (priority 3) - risk-approved orders handed to the broker. They
   can only execute against *later* bars.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar

import pandas as pd

from backtester.orders import OrderType, Side, TimeInForce, validate_order_spec


@dataclass(frozen=True, slots=True)
class Event:
    """Base class of all events."""

    priority: ClassVar[int] = 99
    timestamp: pd.Timestamp


@dataclass(frozen=True, slots=True)
class FillEvent(Event):
    """An execution of (part of) an order."""

    priority: ClassVar[int] = 0
    order_id: int
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    slippage: float = 0.0
    """Cost of slippage in currency units relative to the reference price."""

    @property
    def signed_quantity(self) -> float:
        """Quantity with the sign of the side (positive for buys)."""
        return self.side.sign * self.quantity

    @property
    def notional(self) -> float:
        """Absolute traded value ``quantity * price``."""
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class MarketEvent(Event):
    """A new bar has closed and is now visible to the strategy.

    ``index`` is the position of the bar in the feed's aligned calendar.
    """

    priority: ClassVar[int] = 1
    index: int


@dataclass(frozen=True, slots=True)
class SignalEvent(Event):
    """A strategy's request to trade a specific quantity, pending risk review."""

    priority: ClassVar[int] = 2
    order_id: int
    symbol: str
    quantity: float
    """Signed quantity: positive buys, negative sells."""
    order_type: OrderType = OrderType.MARKET
    tif: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        validate_order_spec(abs(self.quantity), self.order_type, self.limit_price, self.stop_price)


@dataclass(frozen=True, slots=True)
class TargetEvent(Event):
    """A strategy's desired portfolio weights (fraction of equity per symbol).

    With ``partial=False`` (the default) the mapping describes the whole portfolio and
    any symbol not mentioned is targeted to zero. With ``partial=True`` only the listed
    symbols are rebalanced.
    """

    priority: ClassVar[int] = 2
    weights: Mapping[str, float] = field(default_factory=dict)
    partial: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True, slots=True)
class CancelEvent(Event):
    """Cancel one order (``order_id``) or every open order of ``symbol`` (or all)."""

    priority: ClassVar[int] = 2
    order_id: int | None = None
    symbol: str | None = None


@dataclass(frozen=True, slots=True)
class OrderEvent(Event):
    """A risk-approved order ready to be submitted to the broker."""

    priority: ClassVar[int] = 3
    order_id: int
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    tif: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""

    def __post_init__(self) -> None:
        validate_order_spec(self.quantity, self.order_type, self.limit_price, self.stop_price)


class EventQueue:
    """Priority queue of events ordered by ``(timestamp, priority, sequence)``.

    The insertion sequence breaks ties so events of equal timestamp and priority are
    processed first-in, first-out, which keeps runs deterministic.
    """

    __slots__ = ("_heap", "_seq")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, Event]] = []
        self._seq = itertools.count()

    def push(self, event: Event) -> None:
        """Add an event to the queue."""
        key = (event.timestamp.value, event.priority, next(self._seq), event)
        heapq.heappush(self._heap, key)

    def pop(self) -> Event:
        """Remove and return the next event.

        Raises:
            IndexError: if the queue is empty.
        """
        if not self._heap:
            raise IndexError("pop from an empty EventQueue")
        return heapq.heappop(self._heap)[3]

    def peek(self) -> Event | None:
        """Return the next event without removing it, or ``None`` when empty."""
        return self._heap[0][3] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
