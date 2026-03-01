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
    """Supported order types.

    Execution happens on bars after the one the order was decided on:

    * ``MARKET`` - the next bar's opening auction (the open).
    * ``MARKET_ON_OPEN`` - the same; the explicit name for an opening-auction order.
    * ``MARKET_ON_CLOSE`` - the next bar's closing auction (the close). Decided at the
      close of bar ``t``, it fills at the close of ``t + 1``: the close of ``t`` has
      already happened when the decision is made, so filling there would be
      look-ahead.
    * ``LIMIT``, ``STOP``, ``STOP_LIMIT`` - at the open when the bar gaps through the
      price, otherwise where the intrabar path reaches it.
    * ``TRAILING_STOP`` - a stop that trails the best price seen since the order
      became active by ``trail_amount`` (absolute) or ``trail_percent`` (fraction).
    """

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_OPEN = "market_on_open"
    MARKET_ON_CLOSE = "market_on_close"


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


AUCTION_TYPES = frozenset({OrderType.MARKET, OrderType.MARKET_ON_OPEN, OrderType.MARKET_ON_CLOSE})
"""Order types that execute in an auction (the open or the close), not at a price."""
_NEEDS_LIMIT = frozenset({OrderType.LIMIT, OrderType.STOP_LIMIT})
_NEEDS_STOP = frozenset({OrderType.STOP, OrderType.STOP_LIMIT})


def _check_price(name: str, price: float | None, needed: bool, order_type: OrderType) -> None:
    if price is None:
        if needed:
            raise OrderError(f"{order_type.value} orders require {name}")
    elif not needed:
        raise OrderError(f"{order_type.value} orders must not set {name}")
    elif not math.isfinite(price) or price <= 0:
        raise OrderError(f"{name} must be a positive finite number, got {price!r}")


def validate_order_spec(
    quantity: float,
    order_type: OrderType,
    limit_price: float | None,
    stop_price: float | None,
    *,
    trail_amount: float | None = None,
    trail_percent: float | None = None,
) -> None:
    """Validate the static parts of an order specification.

    Raises:
        OrderError: if the quantity is not a positive finite number, if the prices
            required by ``order_type`` are missing, non-positive or superfluous, or if a
            trailing stop does not set exactly one of ``trail_amount`` (``> 0``) and
            ``trail_percent`` (in ``(0, 1)``).
    """
    if not math.isfinite(quantity) or quantity <= 0:
        raise OrderError(f"order quantity must be a positive finite number, got {quantity!r}")
    if (
        order_type in AUCTION_TYPES
        and limit_price is None
        and stop_price is None
        and trail_amount is None
        and trail_percent is None
    ):
        return  # the common case, checked on every order and event
    _check_price("limit_price", limit_price, order_type in _NEEDS_LIMIT, order_type)
    _check_price("stop_price", stop_price, order_type in _NEEDS_STOP, order_type)
    if order_type is OrderType.TRAILING_STOP:
        if (trail_amount is None) == (trail_percent is None):
            raise OrderError("trailing stops need exactly one of trail_amount and trail_percent")
        if trail_amount is not None and not (math.isfinite(trail_amount) and trail_amount > 0):
            raise OrderError(f"trail_amount must be positive, got {trail_amount!r}")
        if trail_percent is not None and not 0 < trail_percent < 1:
            raise OrderError(f"trail_percent must be in (0, 1), got {trail_percent!r}")
    elif trail_amount is not None or trail_percent is not None:
        raise OrderError(f"{order_type.value} orders must not set a trail")


@dataclass(slots=True)
class Order:
    """A working order held by the broker.

    Orders are created by the broker from an approved
    :class:`~backtester.events.OrderEvent`; the broker is the only component that
    mutates them.

    Linked orders: an order with a ``parent_id`` is an exit of a bracket. It is not
    ``armed`` (cannot fill) until its parent - the entry - has filled, and it works for
    the quantity the entry opened. Orders sharing an ``oco_group`` are one-cancels-
    other: the first fill of any of them cancels the rest.
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
    trail_amount: float | None = None
    trail_percent: float | None = None
    trail_reference: float = math.nan
    """Best price since activation: the high for sell trailing stops, the low for buys."""
    parent_id: int | None = None
    oco_group: int | None = None
    armed: bool = True

    def __post_init__(self) -> None:
        validate_order_spec(
            self.quantity,
            self.order_type,
            self.limit_price,
            self.stop_price,
            trail_amount=self.trail_amount,
            trail_percent=self.trail_percent,
        )

    def __copy__(self) -> Order:
        # ``copy.copy`` of a slots dataclass goes through ``__reduce_ex__``; the broker
        # copies every working order of a symbol whose fills depend on the intrabar
        # path, so this spells the copy out (tests check that it covers every field).
        clone = object.__new__(Order)
        clone.id = self.id
        clone.symbol = self.symbol
        clone.side = self.side
        clone.quantity = self.quantity
        clone.order_type = self.order_type
        clone.created_at = self.created_at
        clone.tif = self.tif
        clone.limit_price = self.limit_price
        clone.stop_price = self.stop_price
        clone.tag = self.tag
        clone.filled_quantity = self.filled_quantity
        clone.avg_fill_price = self.avg_fill_price
        clone.status = self.status
        clone.triggered = self.triggered
        clone.closed_at = self.closed_at
        clone.trail_amount = self.trail_amount
        clone.trail_percent = self.trail_percent
        clone.trail_reference = self.trail_reference
        clone.parent_id = self.parent_id
        clone.oco_group = self.oco_group
        clone.armed = self.armed
        return clone

    def trail_level(self, reference: float | None = None) -> float:
        """Stop level of a trailing stop for ``reference`` (default: its own)."""
        ref = self.trail_reference if reference is None else reference
        if self.trail_percent is not None:
            sign = -1.0 if self.side is Side.SELL else 1.0
            return ref * (1.0 + sign * self.trail_percent)
        if self.trail_amount is None:
            raise OrderError(f"order {self.id} is not a trailing stop")
        return ref - self.trail_amount if self.side is Side.SELL else ref + self.trail_amount

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
