"""Simulated broker: order book, linked orders, and matching along an intrabar path."""

from __future__ import annotations

import copy
import math
from collections import ChainMap
from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from backtester.data.feed import Bar, MarketView
from backtester.errors import ConfigError, OrderError
from backtester.events import FillEvent, OrderEvent
from backtester.execution.commission import CommissionModel, NoCommission
from backtester.execution.matching import (
    BarSimulation,
    SimulatedFill,
    idle_on_bar,
    settle_bracket,
)
from backtester.execution.slippage import NoSlippage, SlippageModel
from backtester.orders import Order, OrderStatus, OrderType, TimeInForce

IntrabarPath = Literal["worst", "best", "high_first", "low_first"]
_PATHS = ("worst", "best", "high_first", "low_first")
_INTRABAR = frozenset(
    {OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP}
)


class SimulatedBroker:
    """Matches working orders against OHLCV bars.

    Execution model (all at bar granularity):

    * An order submitted at the close of bar ``t`` first becomes eligible on bar
      ``t + 1``; nothing ever fills on the bar that generated it.
    * **Market** and **market-on-open** orders fill at the open; **market-on-close**
      orders at the close.
    * **Limit** orders fill at the open if the bar gaps through the limit (a better
      price), otherwise at the limit price where the path reaches it.
    * **Stop** orders trigger at the open if the bar gaps through the stop (and fill at
      the open - a worse price), otherwise at the stop price.
    * **Stop-limit** orders become limit orders once triggered; a limit on the far side
      of the stop can still fill later on the same bar if the path comes back to it.
    * **Trailing stops** trail the best price since they became active - the high for
      a sell, the low for a buy - by ``trail_amount`` or ``trail_percent``, ratcheting
      along the intrabar path.
    * **Brackets**: exits (``parent_id`` set) are armed by their entry's first fill,
      work for the quantity the entry opened, reduce each other (a fill of one lowers
      the other by the same amount) and end the entry's unfilled remainder. They can
      fill on the entry's own bar, after the entry on the path. If the entry never
      fills, they are cancelled.
    * **OCO groups**: the first fill of an order of a group cancels the others. All
      members must be orders on the same symbol.
    * Slippage is applied adversely to the reference price; limit-type executions are
      then capped at their limit price.
    * With ``max_participation`` set, the total quantity filled per symbol and bar is
      capped at that fraction of the bar's volume, allocated in path order; the
      remainder stays working (GTC) or expires (DAY).

    **Which intrabar path.** A bar does not say whether the high came before the low.
    For independent limit and stop orders it does not matter. When it does - both exits
    of a bracket inside one bar, OCO orders, trailing stops, stop-limits and contested
    participation capacity - ``intrabar`` decides:

    * ``"worst"`` (default): simulate open-high-low-close and open-low-high-close and
      keep the one whose fills leave the lower equity at the bar's close (the stop, not
      the target, when both could have traded);
    * ``"best"``: the higher equity (optimistic, for sensitivity analysis only);
    * ``"high_first"`` / ``"low_first"``: always that path.

    Args:
        slippage: Slippage model (default: none).
        commission: Commission model (default: none).
        max_participation: Maximum fraction of bar volume the broker may fill per symbol
            and bar, or ``None`` for no cap.
        strict_limits: If true, a limit order needs the price to trade *through* its
            limit (not merely touch it) to fill intrabar - a more conservative
            assumption about queue position.
        intrabar: Path policy, see above.
    """

    def __init__(
        self,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        *,
        max_participation: float | None = None,
        strict_limits: bool = False,
        intrabar: IntrabarPath = "worst",
    ) -> None:
        if max_participation is not None and not 0 < max_participation <= 1:
            raise ConfigError("max_participation must be in (0, 1]")
        if intrabar not in _PATHS:
            raise ConfigError(f"intrabar must be one of {_PATHS}")
        self.slippage: SlippageModel = slippage or NoSlippage()
        self.commission: CommissionModel = commission or NoCommission()
        self.max_participation = max_participation
        self.strict_limits = strict_limits
        self.intrabar: IntrabarPath = intrabar
        self._orders: dict[int, Order] = {}
        self._open: dict[int, Order] = {}
        self._children: dict[int, list[int]] = {}
        self._oco_symbols: dict[int, str] = {}

    # ------------------------------------------------------------------ order book
    @property
    def orders(self) -> list[Order]:
        """Every order ever received, in submission order."""
        return list(self._orders.values())

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """Working orders, optionally filtered by symbol."""
        return [o for o in self._open.values() if symbol is None or o.symbol == symbol]

    def get(self, order_id: int) -> Order:
        """Look up an order by id."""
        return self._orders[order_id]

    def submit(self, event: OrderEvent) -> Order:
        """Accept an approved order. It becomes eligible from the next bar on."""
        if event.order_id in self._orders:
            raise OrderError(f"duplicate order id {event.order_id}")
        if event.parent_id is not None:
            parent = self._orders.get(event.parent_id)
            if parent is None or parent.symbol != event.symbol:
                raise OrderError(
                    f"order {event.order_id}: no entry {event.parent_id} on {event.symbol}"
                )
        if event.oco_group is not None:
            group_symbol = self._oco_symbols.get(event.oco_group, event.symbol)
            if group_symbol != event.symbol:
                raise OrderError("all orders of an OCO group must be on the same symbol")
        reference = math.nan if event.trail_reference is None else float(event.trail_reference)
        order = Order(
            id=event.order_id,
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            order_type=event.order_type,
            created_at=event.timestamp,
            tif=event.tif,
            limit_price=event.limit_price,
            stop_price=event.stop_price,
            tag=event.tag,
            trail_amount=event.trail_amount,
            trail_percent=event.trail_percent,
            trail_reference=reference,
            parent_id=event.parent_id,
            oco_group=event.oco_group,
            armed=event.parent_id is None,
        )
        self._orders[order.id] = order
        self._open[order.id] = order
        if order.parent_id is not None:
            self._children.setdefault(order.parent_id, []).append(order.id)
        if order.oco_group is not None:
            self._oco_symbols.setdefault(order.oco_group, order.symbol)
        return order

    def record_rejection(self, event: OrderEvent) -> Order:
        """Log an order the risk manager refused, for the audit trail."""
        order = self.submit(event)
        self._close(order, OrderStatus.REJECTED, event.timestamp)
        return order

    def cancel(self, order_id: int, timestamp: pd.Timestamp) -> bool:
        """Cancel a working order. Returns ``False`` if it was not open.

        Cancelling a bracket entry before it fills also cancels its exits; after a
        partial fill the exits keep protecting the filled quantity.
        """
        order = self._open.get(order_id)
        if order is None:
            return False
        self._close(order, OrderStatus.CANCELLED, timestamp)
        self._settle(order, timestamp)
        return True

    def cancel_all(self, timestamp: pd.Timestamp, symbol: str | None = None) -> list[Order]:
        """Cancel every working order (of ``symbol`` if given), exits included."""
        cancelled = self.open_orders(symbol)
        for order in cancelled:
            self._close(order, OrderStatus.CANCELLED, timestamp)
        return cancelled

    def _close(self, order: Order, status: OrderStatus, timestamp: pd.Timestamp) -> None:
        order.status = status
        order.closed_at = timestamp
        self._open.pop(order.id, None)

    def _settle(self, parent: Order, timestamp: pd.Timestamp) -> None:
        """Apply the bracket rules after ``parent`` (an entry) stopped working."""
        if parent.id not in self._children:
            return
        settle_bracket(parent, self._children, self.get)
        for child_id in self._children[parent.id]:
            self._sync(self._orders[child_id], timestamp)

    def _sync(self, order: Order, timestamp: pd.Timestamp) -> None:
        """Take an order that reached a terminal state out of the open book."""
        if order.status.is_terminal and order.id in self._open:
            order.closed_at = timestamp
            del self._open[order.id]

    # ------------------------------------------------------------------ matching
    def process_bar(self, view: MarketView) -> list[FillEvent]:
        """Match working orders against the bar at ``view.timestamp``.

        Returns the fills in the order they happened on the bar: by path time (open,
        intrabar, close), then by order id.
        """
        ts = view.timestamp
        by_symbol: dict[str, list[Order]] = {}
        for order_id in sorted(self._open):
            order = self._open[order_id]
            if order.created_at < ts:
                by_symbol.setdefault(order.symbol, []).append(order)
        timed: list[tuple[float, int, FillEvent]] = []
        for symbol, orders in by_symbol.items():
            bar = view.bar(symbol)
            if bar is not None:
                for fill in self._match(orders, bar):
                    event = FillEvent(
                        timestamp=ts,
                        order_id=fill.order_id,
                        symbol=symbol,
                        side=self._orders[fill.order_id].side,
                        quantity=fill.quantity,
                        price=fill.price,
                        commission=fill.commission,
                        slippage=fill.slippage,
                    )
                    timed.append((fill.time, fill.order_id, event))
            self._end_of_bar([o.id for o in orders], ts)
        timed.sort(key=lambda item: (item[0], item[1]))
        return [event for _, _, event in timed]

    def _capacity(self, bar: Bar) -> float:
        if self.max_participation is None:
            return math.inf
        return float(math.floor(self.max_participation * bar.volume))

    def _path_sensitive(self, orders: Sequence[Order], capacity: float) -> bool:
        """Whether the intrabar path can change this symbol's fills on this bar."""
        intrabar = 0
        for order in orders:
            if (
                order.order_type in (OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)
                or order.parent_id is not None
                or order.oco_group is not None
                or order.id in self._children
            ):
                return True
            intrabar += order.order_type in _INTRABAR
        return math.isfinite(capacity) and intrabar > 1

    def _simulate(
        self,
        orders: Sequence[Order],
        bar: Bar,
        capacity: float,
        *,
        high_first: bool,
        copies: bool,
    ) -> BarSimulation:
        """Run one path. With ``copies``, ``orders`` stand in for their book originals."""
        book: Mapping[int, Order] = self._orders
        if copies:
            book = ChainMap({o.id: o for o in orders}, self._orders)
        sim = BarSimulation(
            orders,
            book.__getitem__,
            self._children,
            bar,
            high_first=high_first,
            capacity=capacity,
            slippage=self.slippage,
            commission=self.commission,
            strict_limits=self.strict_limits,
        )
        sim.run()
        return sim

    def _match(self, orders: list[Order], bar: Bar) -> list[SimulatedFill]:
        if all(idle_on_bar(order, bar) for order in orders):
            return []  # nothing can execute, so nothing can change, on either path
        capacity = self._capacity(bar)
        if self.intrabar in ("high_first", "low_first") or not self._path_sensitive(
            orders, capacity
        ):
            high_first = self.intrabar != "low_first"
            sim = self._simulate(orders, bar, capacity, high_first=high_first, copies=False)
            return sim.fills
        # Compare both paths on copies, then install the chosen copies in the book.
        candidates = []
        for high_first in (True, False):
            copies = [copy.copy(o) for o in orders]
            sim = self._simulate(copies, bar, capacity, high_first=high_first, copies=True)
            candidates.append((copies, sim))
        values = [sim.close_value() for _, sim in candidates]
        worse_second = values[1] < values[0]
        pick_second = worse_second if self.intrabar == "worst" else values[1] > values[0]
        copies, chosen = candidates[1] if pick_second else candidates[0]
        for replacement in copies:
            self._orders[replacement.id] = replacement
            if replacement.id in self._open:
                self._open[replacement.id] = replacement
        return chosen.fills

    def _end_of_bar(self, order_ids: list[int], timestamp: pd.Timestamp) -> None:
        """Close what finished on this bar, expire DAY orders, settle brackets."""
        for oid in order_ids:
            order = self._orders[oid]
            if not order.status.is_terminal and order.armed and order.tif is TimeInForce.DAY:
                order.status = OrderStatus.EXPIRED
            if order.status.is_terminal:
                self._sync(order, timestamp)
        for oid in order_ids:
            order = self._orders[oid]
            if order.status.is_terminal and oid in self._children:
                self._settle(order, timestamp)
