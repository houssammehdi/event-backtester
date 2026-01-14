"""Order domain model: sides, order types, time-in-force and the mutable order record."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, StrEnum

import pandas as pd

from backtester.errors import OrderError


class Side(Enum):
    """Direction of an order or fill."""

    BUY = 1
    SELL = -1

    @property
    def sign(self) -> int:
        """``+1`` for buys and ``-1`` for sells."""
        return self.value

    @classmethod
    def from_quantity(cls, quantity: float) -> Side:
        """Return the side implied by a signed quantity (positive means buy)."""
        if quantity == 0:
            raise OrderError("cannot infer a side from a zero quantity")
        return cls.BUY if quantity > 0 else cls.SELL


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    """How long an unfilled order keeps working.

    ``DAY`` orders are valid for the first bar after submission only; any unfilled
    remainder expires at that bar's close. ``GTC`` orders keep working until they are
    filled or cancelled.
    """

    DAY = "day"
    GTC = "gtc"


class OrderStatus(StrEnum):
    """Lifecycle state of an order."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        """Whether no further fills can happen in this state."""
        return self in _TERMINAL


_TERMINAL = frozenset(
    {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
)


def validate_order_spec(
    quantity: float,
    order_type: OrderType,
    limit_price: float | None,
    stop_price: float | None,
) -> None:
    """Validate the static parts of an order specification.

    Raises:
        OrderError: if the quantity is not a positive finite number, or if the prices
            required by ``order_type`` are missing, non-positive or superfluous.
    """
    if not math.isfinite(quantity) or quantity <= 0:
        raise OrderError(f"order quantity must be a positive finite number, got {quantity!r}")
    needs_limit = order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
    needs_stop = order_type in (OrderType.STOP, OrderType.STOP_LIMIT)
    for name, price, needed in (
        ("limit_price", limit_price, needs_limit),
        ("stop_price", stop_price, needs_stop),
    ):
        if needed and price is None:
            raise OrderError(f"{order_type.value} orders require {name}")
        if not needed and price is not None:
            raise OrderError(f"{order_type.value} orders must not set {name}")
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise OrderError(f"{name} must be a positive finite number, got {price!r}")


@dataclass(slots=True)
class Order:
    """A working order held by the broker.

    Orders are created by the broker from an approved
    :class:`~backtester.events.OrderEvent`; the broker is the only component that
    mutates them.
    """

    id: int
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    created_at: pd.Timestamp
    tif: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str = ""
    filled_quantity: float = 0.0
    avg_fill_price: float = math.nan
    status: OrderStatus = OrderStatus.NEW
    triggered: bool = False
    closed_at: pd.Timestamp | None = field(default=None)

    def __post_init__(self) -> None:
        validate_order_spec(self.quantity, self.order_type, self.limit_price, self.stop_price)

    @property
    def remaining(self) -> float:
        """Quantity still to be filled."""
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def is_active(self) -> bool:
        """Whether the order can still receive fills."""
        return not self.status.is_terminal

    @property
    def signed_remaining(self) -> float:
        """Remaining quantity with the sign of the order side."""
        return self.side.sign * self.remaining

    def record_fill(self, quantity: float, price: float) -> None:
        """Apply a (partial) fill to the order's bookkeeping."""
        if quantity <= 0 or quantity > self.remaining + 1e-9:
            raise OrderError(f"invalid fill quantity {quantity} for order {self.id}")
        previous = self.filled_quantity
        self.filled_quantity = previous + quantity
        if previous == 0:
            self.avg_fill_price = price
        else:
            self.avg_fill_price = (
                self.avg_fill_price * previous + price * quantity
            ) / self.filled_quantity
        if self.remaining <= 1e-9:
            self.filled_quantity = self.quantity
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
