"""Matching one symbol's working orders against one bar along an explicit intrabar path.

A bar only records ``open, high, low, close``; the order in which prices traded is
unknown. The simulation walks one of the two canonical paths through those points,

* ``high_first``: open -> high -> low -> close;
* ``low_first``: open -> low -> high -> close;

and executes orders in the order the path reaches them:

1. **Opening auction** (time 0) at the open: market and market-on-open orders, and
   every limit, stop, stop-limit or trailing stop the open already satisfies (gaps
   fill at the open). Exits armed by a fill here are checked at the open too.
2. **Intrabar**, one monotone segment at a time (segment ``k`` spans times ``k`` to
   ``k + 1``): limits and stops execute at their price, trailing stops at their
   trailing level, and stop-limits trigger at their stop and then work as limits for
   the rest of the path. An order armed or triggered mid-segment is live from that
   point of the path onwards.
3. **Closing auction** (time 3.5) at the close: market-on-close orders.

For independent limit and stop orders the path does not matter: both paths visit
every price between the low and the high, so they fill at the same prices. It matters
for linked orders (brackets, OCO), trailing stops (the trailing level ratchets up as
the path makes new highs), stop-limits (whether the limit is revisited after the
trigger) and for which orders get a scarce participation capacity first. The broker
decides which path to use (see :class:`backtester.execution.SimulatedBroker`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from backtester.data.feed import Bar
from backtester.execution.commission import CommissionModel
from backtester.execution.slippage import SlippageModel
from backtester.orders import Order, OrderStatus, OrderType, Side

OPEN_TIME = 0.0
CLOSE_TIME = 3.5
_OPEN_ONLY = frozenset({OrderType.MARKET, OrderType.MARKET_ON_OPEN})
_NOT_INTRABAR = _OPEN_ONLY | {OrderType.MARKET_ON_CLOSE}
_EPS = 1e-9

Lookup = Callable[[int], Order]


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """One execution produced by :class:`BarSimulation`."""

    time: float
    """Position on the path: 0 = open, 0..3 = the three segments, 3.5 = close."""
    order_id: int
    quantity: float
    price: float
    commission: float
    slippage: float
    """Slippage cost in currency (price distance from the reference times quantity)."""


@dataclass(frozen=True, slots=True)
class _Hit:
    time: float
    order: Order
    at: float
    """Path price where the event happens."""
    reference: float
    """Execution price before slippage."""
    limit: float | None
    trigger_only: bool = False
    """A stop-limit trigger that does not fill yet (its limit is not reachable here)."""


def _crosses(buy: bool, up: bool, position: float, stop: float, end: float) -> bool:
    """A buy stop is crossed on the way up, a sell stop on the way down."""
    if buy:
        return up and position < stop <= end
    return not up and end <= stop < position


def path_points(bar: Bar, high_first: bool) -> tuple[float, float, float, float]:
    """``(open, high, low, close)`` or ``(open, low, high, close)``."""
    if high_first:
        return bar.open, bar.high, bar.low, bar.close
    return bar.open, bar.low, bar.high, bar.close


def open_quantity(parent: Order, children: Mapping[int, list[int]], lookup: Lookup) -> float:
    """Quantity a bracket's entry opened that its exits have not closed yet."""
    closed = sum(lookup(c).filled_quantity for c in children.get(parent.id, ()))
    return max(parent.filled_quantity - closed, 0.0)


def settle_bracket(parent: Order, children: Mapping[int, list[int]], lookup: Lookup) -> None:
    """Close the exits of a bracket whose entry is finished and whose position is flat.

    An exit that closed part of the position is resized to what it executed and marked
    filled; the others are cancelled. Nothing happens while the entry can still fill
    or some of the opened quantity is still to be closed.
    """
    if not parent.status.is_terminal or open_quantity(parent, children, lookup) > _EPS:
        return
    for child_id in children.get(parent.id, ()):
        child = lookup(child_id)
        if child.status.is_terminal:
            continue
        if child.filled_quantity > 0:
            child.quantity = child.filled_quantity
            child.status = OrderStatus.FILLED
        else:
            child.status = OrderStatus.CANCELLED


class BarSimulation:
    """Execute the working orders of one symbol on one bar, along one path.

    The simulation mutates the given :class:`~backtester.orders.Order` objects - fills,
    trigger flags, trailing references, arming of exits, cancellations - so the broker
    runs it on copies when it has to compare paths.

    Args:
        orders: The symbol's working orders that are eligible on this bar (id order).
        lookup: Finds any order of the book by id, including terminal ones - used for
            bracket bookkeeping (an entry that filled on an earlier bar).
        children: Ids of each bracket entry's exits.
        bar: The bar.
        high_first: Which canonical path to walk.
        capacity: Quantity the participation cap allows on this bar.
        slippage: Slippage model.
        commission: Commission model.
        strict_limits: Limits need the price to trade through them intrabar.
    """

    def __init__(
        self,
        orders: Sequence[Order],
        lookup: Lookup,
        children: Mapping[int, list[int]],
        bar: Bar,
        *,
        high_first: bool,
        capacity: float,
        slippage: SlippageModel,
        commission: CommissionModel,
        strict_limits: bool,
    ) -> None:
        self.orders = list(orders)
        self._lookup = lookup
        self._children = children
        self.bar = bar
        self.points = path_points(bar, high_first)
        self.capacity = capacity
        self._slippage = slippage
        self._commission = commission
        self._strict = strict_limits
        self._blocked: set[int] = set()
        self.fills: list[SimulatedFill] = []

    # ------------------------------------------------------------------ driver
    def run(self) -> list[SimulatedFill]:
        """Walk the path; returns the fills in path order."""
        self._auction(self.points[0], OPEN_TIME, closing=False)
        for k in range(3):
            a, b = self.points[k], self.points[k + 1]
            if a != b:
                self._segment(k, a, b)
            self._update_trailing(a, b)
        self._auction(self.points[3], CLOSE_TIME, closing=True)
        return self.fills

    def close_value(self) -> float:
        """Mark-to-close value of this bar's fills net of commission.

        ``sum(sign * quantity * (close - price)) - commission``: how much the fills
        added to equity at the bar's close. The worst-case policy minimises it.
        """
        close = self.bar.close
        side = {o.id: o.side.sign for o in self.orders}
        return sum(
            side[f.order_id] * f.quantity * (close - f.price) - f.commission for f in self.fills
        )

    # ------------------------------------------------------------------ state
    def _live(self) -> list[Order]:
        return [
            o
            for o in self.orders
            if o.armed and not o.status.is_terminal and o.id not in self._blocked
        ]

    def _fillable(self, order: Order) -> float:
        quantity = order.remaining
        if order.parent_id is not None:
            parent = self._lookup(order.parent_id)
            quantity = min(quantity, open_quantity(parent, self._children, self._lookup))
        return min(quantity, self.capacity)

    # ------------------------------------------------------------------ matching
    def _auction(self, price: float, time: float, *, closing: bool) -> None:
        progress = True
        while progress:  # a fill can arm exits that the same price already triggers
            progress = False
            for order in self._live():
                if closing:
                    if order.order_type is OrderType.MARKET_ON_CLOSE and self._execute(
                        order, time, price, None
                    ):
                        progress = True
                    continue
                if order.order_type is OrderType.MARKET_ON_CLOSE:
                    continue
                if order.order_type is OrderType.TRAILING_STOP:
                    self._follow(order, price)
                hit = self._marketable(order, price, time, opening=True)
                if hit is None:
                    continue
                if hit.trigger_only:
                    order.triggered = True
                elif self._execute(order, time, hit.reference, hit.limit):
                    progress = True

    def _segment(self, k: int, a: float, b: float) -> None:
        up = b > a
        position = a
        while True:
            best: _Hit | None = None
            for order in self._live():
                if order.order_type in _NOT_INTRABAR:
                    continue
                hit = self._reach(order, position, a, b, up, k)
                if hit is not None and (
                    best is None or (hit.time, hit.order.id) < (best.time, best.order.id)
                ):
                    best = hit
            if best is None:
                return
            position = best.at
            if best.trigger_only:
                best.order.triggered = True
            else:
                self._execute(best.order, best.time, best.reference, best.limit)

    @staticmethod
    def _time(k: int, a: float, b: float, at: float) -> float:
        return k + abs(at - a) / abs(b - a)

    def _marketable(self, order: Order, price: float, time: float, *, opening: bool) -> _Hit | None:
        """The order is executable (or, for a stop-limit, triggered) at ``price`` now."""
        kind, buy = order.order_type, order.side is Side.BUY
        if kind in _OPEN_ONLY:
            return _Hit(time, order, price, price, None) if opening else None
        if kind is OrderType.LIMIT:
            assert order.limit_price is not None
            return self._limit_now(order, order.limit_price, price, time, opening=opening)
        if kind is OrderType.STOP or kind is OrderType.TRAILING_STOP:
            stop = order.trail_level() if kind is OrderType.TRAILING_STOP else order.stop_price
            assert stop is not None
            crossed = price >= stop if buy else price <= stop
            return _Hit(time, order, price, price, None) if crossed else None
        assert order.limit_price is not None
        assert order.stop_price is not None
        if not order.triggered and not (
            price >= order.stop_price if buy else price <= order.stop_price
        ):
            return None
        return self._limit_now(order, order.limit_price, price, time, opening=True) or _Hit(
            time, order, price, price, None, trigger_only=True
        )

    def _limit_now(
        self, order: Order, limit: float, price: float, time: float, *, opening: bool
    ) -> _Hit | None:
        # Touching the limit counts at the open (the auction prints there); intrabar,
        # strict_limits require trading through it.
        strict = self._strict and not opening
        if order.side is Side.BUY:
            ok = price < limit if strict else price <= limit
        else:
            ok = price > limit if strict else price >= limit
        return _Hit(time, order, price, price, limit) if ok else None

    def _reach(
        self, order: Order, position: float, a: float, b: float, up: bool, k: int
    ) -> _Hit | None:
        """Where the order executes as the price moves from ``position`` to ``b``."""
        now = self._marketable(order, position, self._time(k, a, b, position), opening=False)
        if now is not None and not (now.trigger_only and order.triggered):
            return now
        kind, buy = order.order_type, order.side is Side.BUY
        if kind is OrderType.LIMIT or (kind is OrderType.STOP_LIMIT and order.triggered):
            assert order.limit_price is not None
            limit = order.limit_price
            if buy and not up and limit < position:  # a buy limit is reached going down
                reached = b < limit if self._strict else b <= limit
            elif not buy and up and limit > position:  # a sell limit going up
                reached = b > limit if self._strict else b >= limit
            else:
                reached = False
            return _Hit(self._time(k, a, b, limit), order, limit, limit, limit) if reached else None
        if kind is OrderType.STOP or kind is OrderType.TRAILING_STOP:
            stop = order.trail_level() if kind is OrderType.TRAILING_STOP else order.stop_price
            assert stop is not None
            if _crosses(buy, up, position, stop, b):
                return _Hit(self._time(k, a, b, stop), order, stop, stop, None)
            return None
        # untriggered stop-limit: triggers where the path crosses its stop
        assert order.stop_price is not None
        assert order.limit_price is not None
        stop, limit = order.stop_price, order.limit_price
        if not _crosses(buy, up, position, stop, b):
            return None
        marketable = limit >= stop if buy else limit <= stop
        return _Hit(
            self._time(k, a, b, stop), order, stop, stop, limit, trigger_only=not marketable
        )

    # ------------------------------------------------------------------ execution
    def _execute(self, order: Order, time: float, reference: float, limit: float | None) -> bool:
        quantity = self._fillable(order)
        if quantity <= 0:
            self._blocked.add(order.id)
            return False
        price = self._slippage.fill_price(reference, order.side, quantity, self.bar)
        if limit is not None:
            price = min(price, limit) if order.side is Side.BUY else max(price, limit)
        commission = self._commission.commission(quantity, price)
        order.record_fill(quantity, price)
        self.capacity -= quantity
        self.fills.append(
            SimulatedFill(
                time=time,
                order_id=order.id,
                quantity=quantity,
                price=price,
                commission=commission,
                slippage=abs(price - reference) * quantity,
            )
        )
        self._link_effects(order, reference)
        return True

    def _link_effects(self, order: Order, price: float) -> None:
        """OCO cancellation and bracket arming after a fill at market price ``price``."""
        if order.oco_group is not None:  # one cancels the others
            for other in self.orders:
                same_group = other is not order and other.oco_group == order.oco_group
                if same_group and not other.status.is_terminal:
                    other.status = OrderStatus.CANCELLED
        for child_id in self._children.get(order.id, ()):  # an entry fill arms its exits
            child = self._lookup(child_id)
            if not child.armed and not child.status.is_terminal:
                child.armed = True
                if child.order_type is OrderType.TRAILING_STOP:
                    child.trail_reference = price
        if order.parent_id is not None:  # an exit fill ends the entry
            parent = self._lookup(order.parent_id)
            if not parent.status.is_terminal:
                parent.status = OrderStatus.CANCELLED
            settle_bracket(parent, self._children, self._lookup)

    @staticmethod
    def _follow(order: Order, price: float) -> None:
        """Let a trailing stop's best price follow ``price`` (starting there if unset)."""
        reference = order.trail_reference
        if math.isnan(reference):
            order.trail_reference = price
        elif order.side is Side.SELL:
            order.trail_reference = max(reference, price)
        else:
            order.trail_reference = min(reference, price)

    def _update_trailing(self, a: float, b: float) -> None:
        for order in self._live():
            if order.order_type is OrderType.TRAILING_STOP:
                self._follow(order, max(a, b) if order.side is Side.SELL else min(a, b))
