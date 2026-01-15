"""Simulated broker: order book, bar-level matching, slippage, commission, participation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from backtester.data.feed import Bar, MarketView
from backtester.errors import ConfigError, OrderError
from backtester.events import FillEvent, OrderEvent
from backtester.execution.commission import CommissionModel, NoCommission
from backtester.execution.slippage import NoSlippage, SlippageModel
from backtester.orders import Order, OrderStatus, OrderType, Side, TimeInForce


@dataclass(frozen=True, slots=True)
class _Match:
    """Where an order would execute on a bar, before slippage and sizing."""

    reference: float
    """Price before slippage."""
    limit: float | None
    """Best acceptable price after slippage (limit-type executions), if any."""


class SimulatedBroker:
    """Matches working orders against OHLCV bars.

    Execution model (all at bar granularity):

    * An order submitted at the close of bar ``t`` first becomes eligible on bar
      ``t + 1``; nothing ever fills on the bar that generated it.
    * **Market** orders fill at the open.
    * **Limit** orders fill at the open if the bar gaps through the limit (a better
      price), otherwise at the limit price if the bar's range reaches it.
    * **Stop** orders trigger at the open if the bar gaps through the stop (and fill at
      the open - a worse price), otherwise at the stop price if the range reaches it.
    * **Stop-limit** orders become limit orders once triggered. If the stop triggers
      intrabar and the limit is on the far side of the stop, the fill is deferred to
      later bars because the intrabar path after the trigger is unknown.
    * Slippage is applied adversely to the reference price; limit-type executions are
      then capped at their limit price.
    * With ``max_participation`` set, the total quantity filled per symbol and bar is
      capped at that fraction of the bar's volume; the remainder stays working (GTC)
      or expires (DAY).

    Args:
        slippage: Slippage model (default: none).
        commission: Commission model (default: none).
        max_participation: Maximum fraction of bar volume the broker may fill per symbol
            and bar, or ``None`` for no cap.
        strict_limits: If true, a limit order needs the price to trade *through* its
            limit (not merely touch it) to fill intrabar - a more conservative
            assumption about queue position.
    """

    def __init__(
        self,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        *,
        max_participation: float | None = None,
        strict_limits: bool = False,
    ) -> None:
        if max_participation is not None and not 0 < max_participation <= 1:
            raise ConfigError("max_participation must be in (0, 1]")
        self.slippage: SlippageModel = slippage or NoSlippage()
        self.commission: CommissionModel = commission or NoCommission()
        self.max_participation = max_participation
        self.strict_limits = strict_limits
        self._orders: dict[int, Order] = {}
        self._open: dict[int, Order] = {}

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
        )
        self._orders[order.id] = order
        self._open[order.id] = order
        return order

    def record_rejection(self, event: OrderEvent) -> Order:
        """Log an order the risk manager refused, for the audit trail."""
        order = self.submit(event)
        self._close(order, OrderStatus.REJECTED, event.timestamp)
        return order

    def cancel(self, order_id: int, timestamp: pd.Timestamp) -> bool:
        """Cancel a working order. Returns ``False`` if it was not open."""
        order = self._open.get(order_id)
        if order is None:
            return False
        self._close(order, OrderStatus.CANCELLED, timestamp)
        return True

    def cancel_all(self, timestamp: pd.Timestamp, symbol: str | None = None) -> list[Order]:
        """Cancel every working order (of ``symbol`` if given)."""
        cancelled = self.open_orders(symbol)
        for order in cancelled:
            self._close(order, OrderStatus.CANCELLED, timestamp)
        return cancelled

    def _close(self, order: Order, status: OrderStatus, timestamp: pd.Timestamp) -> None:
        order.status = status
        order.closed_at = timestamp
        self._open.pop(order.id, None)

    # ------------------------------------------------------------------ matching
    def process_bar(self, view: MarketView) -> list[FillEvent]:
        """Match working orders against the bar at ``view.timestamp``.

        Returns the resulting fills in the order they were generated (order id order).
        """
        ts = view.timestamp
        fills: list[FillEvent] = []
        used: dict[str, float] = {}
        for order in sorted(self._open.values(), key=lambda o: o.id):
            if order.created_at >= ts:
                continue
            bar = view.bar(order.symbol)
            if bar is not None:
                fill = self._try_fill(order, bar, used)
                if fill is not None:
                    fills.append(fill)
            if order.status is OrderStatus.FILLED:
                self._close(order, OrderStatus.FILLED, ts)
            elif order.tif is TimeInForce.DAY:
                self._close(order, OrderStatus.EXPIRED, ts)
        return fills

    def _capacity(self, bar: Bar, used: dict[str, float]) -> float:
        if self.max_participation is None:
            return math.inf
        cap = math.floor(self.max_participation * bar.volume)
        return max(cap - used.get(bar.symbol, 0.0), 0.0)

    def _try_fill(self, order: Order, bar: Bar, used: dict[str, float]) -> FillEvent | None:
        match = self._match(order, bar)
        if match is None:
            return None
        quantity = min(order.remaining, self._capacity(bar, used))
        if quantity <= 0:
            return None
        price = self.slippage.fill_price(match.reference, order.side, quantity, bar)
        if match.limit is not None:
            price = min(price, match.limit) if order.side is Side.BUY else max(price, match.limit)
        commission = self.commission.commission(quantity, price)
        order.record_fill(quantity, price)
        used[bar.symbol] = used.get(bar.symbol, 0.0) + quantity
        return FillEvent(
            timestamp=bar.timestamp,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=abs(price - match.reference) * quantity,
        )

    def _match(self, order: Order, bar: Bar) -> _Match | None:
        kind = order.order_type
        if kind is OrderType.MARKET:
            return _Match(bar.open, None)
        if kind is OrderType.LIMIT:
            assert order.limit_price is not None
            return self._match_limit(order.side, order.limit_price, bar)
        if kind is OrderType.STOP:
            assert order.stop_price is not None
            return self._match_stop(order.side, order.stop_price, bar)
        return self._match_stop_limit(order, bar)

    def _match_limit(self, side: Side, limit: float, bar: Bar) -> _Match | None:
        if side is Side.BUY:
            if bar.open <= limit:
                return _Match(bar.open, limit)
            reached = bar.low < limit if self.strict_limits else bar.low <= limit
        else:
            if bar.open >= limit:
                return _Match(bar.open, limit)
            reached = bar.high > limit if self.strict_limits else bar.high >= limit
        return _Match(limit, limit) if reached else None

    @staticmethod
    def _stop_trigger(side: Side, stop: float, bar: Bar) -> tuple[bool, bool]:
        """Return ``(triggered_at_open, triggered_intrabar)``."""
        if side is Side.BUY:
            return bar.open >= stop, bar.high >= stop
        return bar.open <= stop, bar.low <= stop

    def _match_stop(self, side: Side, stop: float, bar: Bar) -> _Match | None:
        at_open, intrabar = self._stop_trigger(side, stop, bar)
        if at_open:
            return _Match(bar.open, None)
        if intrabar:
            return _Match(stop, None)
        return None

    def _match_stop_limit(self, order: Order, bar: Bar) -> _Match | None:
        assert order.stop_price is not None
        assert order.limit_price is not None
        stop, limit, side = order.stop_price, order.limit_price, order.side
        if order.triggered:
            return self._match_limit(side, limit, bar)
        at_open, intrabar = self._stop_trigger(side, stop, bar)
        if at_open:
            order.triggered = True
            return self._match_limit(side, limit, bar)
        if not intrabar:
            return None
        order.triggered = True
        # Triggered at the stop price mid-bar: marketable immediately only if the limit
        # is at or beyond the stop; otherwise the post-trigger path is unknown.
        marketable = limit >= stop if side is Side.BUY else limit <= stop
        return _Match(stop, limit) if marketable else None
