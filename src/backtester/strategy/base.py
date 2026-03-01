"""Strategy interface and the context object strategies use to act."""

from __future__ import annotations

import copy
import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from backtester.data.feed import DataFeed, MarketView
from backtester.errors import OrderError
from backtester.events import CancelEvent, Event, FillEvent, SignalEvent, TargetEvent
from backtester.execution.broker import SimulatedBroker
from backtester.orders import Order, OrderType, TimeInForce
from backtester.portfolio.portfolio import Portfolio


@dataclass(frozen=True, slots=True)
class Bracket:
    """Order ids of a bracket (see :meth:`StrategyContext.bracket`)."""

    entry: int
    stop_loss: int | None
    """Protective stop (fixed or trailing), if any."""
    take_profit: int | None
    """Profit target, if any."""


class StrategyContext:
    """The strategy's handle on its account and on the order flow.

    Read methods expose account state (cash, equity, positions, working orders).
    Action methods do not act immediately: they record intents - a
    :class:`~backtester.events.TargetEvent`, :class:`~backtester.events.SignalEvent` or
    :class:`~backtester.events.CancelEvent` - that the engine pushes through the event
    queue to the risk manager and broker after ``on_bar`` returns.

    Args:
        portfolio: The account the strategy trades.
        broker: The broker holding the working orders.
        next_id: Order-id generator shared with the engine.
        symbols: The tradable universe. When given, action methods raise
            :class:`~backtester.errors.OrderError` for any other symbol at the call
            site, instead of the mistake surfacing (or vanishing) later.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        broker: SimulatedBroker,
        next_id: Callable[[], int] | None = None,
        *,
        symbols: Iterable[str] | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._broker = broker
        self._next_id = next_id or itertools.count(1).__next__
        self._universe: frozenset[str] | None = None if symbols is None else frozenset(symbols)
        self._events: list[Event] = []
        self._timestamp: pd.Timestamp | None = None
        self._muted = False

    def _check_symbol(self, symbol: str) -> None:
        if self._universe is not None and symbol not in self._universe:
            raise OrderError(f"unknown symbol {symbol!r}: it is not in the data feed")

    # ------------------------------------------------------------------ engine hooks
    def _begin(self, timestamp: pd.Timestamp, *, muted: bool = False) -> None:
        self._timestamp = timestamp
        self._muted = muted

    def _drain(self) -> list[Event]:
        events, self._events = self._events, []
        return [] if self._muted else events

    @property
    def timestamp(self) -> pd.Timestamp:
        """Timestamp of the bar being processed."""
        if self._timestamp is None:
            raise RuntimeError("context is not attached to a running backtest")
        return self._timestamp

    @property
    def is_warmup(self) -> bool:
        """True while bars are replayed before the trading window (intents dropped)."""
        return self._muted

    # ------------------------------------------------------------------ account state
    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._portfolio.cash

    @property
    def equity(self) -> float:
        """Current marked-to-market equity."""
        return self._portfolio.equity

    def position(self, symbol: str) -> float:
        """Signed quantity held in ``symbol``."""
        return self._portfolio.quantity(symbol)

    def positions(self) -> dict[str, float]:
        """Non-zero signed quantities by symbol."""
        return {s: p.quantity for s, p in self._portfolio.positions.items() if not p.is_flat}

    def weights(self) -> dict[str, float]:
        """Current signed position weights."""
        return self._portfolio.weights()

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """Snapshots (copies) of working orders."""
        return [copy.copy(o) for o in self._broker.open_orders(symbol)]

    # ------------------------------------------------------------------ actions
    def target_weights(self, weights: Mapping[str, float], *, partial: bool = False) -> None:
        """Request a rebalance to ``weights`` (fractions of equity) at the next open.

        With ``partial=False`` symbols not listed are closed and every working order is
        cancelled; with ``partial=True`` only the listed symbols are touched.
        """
        for symbol in weights:
            self._check_symbol(symbol)
        self._events.append(TargetEvent(self.timestamp, weights=weights, partial=partial))

    def order(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_amount: float | None = None,
        trail_percent: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
        oco_group: int | None = None,
        tag: str = "",
    ) -> int:
        """Request an order for a signed ``quantity``; returns the order id.

        Args:
            symbol: Symbol to trade.
            quantity: Signed quantity (positive buys, negative sells).
            order_type: See :class:`~backtester.orders.OrderType`.
            limit_price: Limit of ``LIMIT`` and ``STOP_LIMIT`` orders.
            stop_price: Stop of ``STOP`` and ``STOP_LIMIT`` orders.
            trail_amount: Absolute trail of a ``TRAILING_STOP`` (e.g. ``2.5``).
            trail_percent: Relative trail of a ``TRAILING_STOP`` (e.g. ``0.05``); it
                starts trailing from this bar's close.
            tif: Time in force.
            oco_group: Id from :meth:`oco_group`; the first fill in a group cancels the
                other orders of the group.
            tag: Free-form label kept on the order.
        """
        self._check_symbol(symbol)
        order_id = self._next_id()
        self._events.append(
            SignalEvent(
                self.timestamp,
                order_id=order_id,
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                tif=tif,
                limit_price=limit_price,
                stop_price=stop_price,
                tag=tag,
                trail_amount=trail_amount,
                trail_percent=trail_percent,
                oco_group=oco_group,
            )
        )
        return order_id

    def oco_group(self) -> int:
        """A new one-cancels-other group id to pass to :meth:`order`."""
        return self._next_id()

    def bracket(
        self,
        symbol: str,
        quantity: float,
        entry_type: OrderType = OrderType.MARKET,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trail_amount: float | None = None,
        trail_percent: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
        exit_tif: TimeInForce = TimeInForce.GTC,
        tag: str = "",
    ) -> Bracket:
        """Request an entry with a protective stop and/or a profit target.

        The exits are armed when the entry fills and work for the quantity it opened;
        the first exit to fill reduces the other by the same amount and ends any
        unfilled part of the entry; if the entry never fills, both are cancelled. When
        the stop and the target both trade inside one bar, the broker's ``intrabar``
        policy decides which came first (by default the stop: the worst case).

        Args:
            symbol: Symbol to trade.
            quantity: Signed entry quantity (positive opens a long).
            entry_type: Order type of the entry (default market).
            limit_price: Entry limit (limit and stop-limit entries).
            stop_price: Entry stop (stop and stop-limit entries).
            stop_loss: Stop price of the protective exit (below the entry for a long).
            take_profit: Limit price of the profit target (above the entry for a long).
            trail_amount: Make the protective exit a trailing stop with this trail...
            trail_percent: ...or with this relative trail, starting from the entry's
                fill price.
            tif: Time in force of the entry.
            exit_tif: Time in force of the exits once armed (default GTC).
            tag: Label of the entry; the exits get ``stop_loss`` / ``take_profit``.

        Returns:
            The order ids of the entry and the exits.
        """
        trailing = trail_amount is not None or trail_percent is not None
        if stop_loss is None and take_profit is None and not trailing:
            raise OrderError("a bracket needs a stop_loss, a trail or a take_profit")
        if stop_loss is not None and trailing:
            raise OrderError("give either a fixed stop_loss or a trail, not both")
        if stop_loss is not None and take_profit is not None:
            ordered = stop_loss < take_profit if quantity > 0 else stop_loss > take_profit
            if not ordered:
                side = "below" if quantity > 0 else "above"
                raise OrderError(f"stop_loss must be {side} take_profit")
        entry = self.order(
            symbol,
            quantity,
            entry_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            tag=tag,
        )
        prefix = f"{tag}:" if tag else ""
        protect: int | None = None
        if stop_loss is not None or trailing:
            protect = self._exit(
                symbol,
                -quantity,
                OrderType.STOP if stop_loss is not None else OrderType.TRAILING_STOP,
                entry,
                exit_tif,
                f"{prefix}stop_loss",
                stop_price=stop_loss,
                trail_amount=trail_amount,
                trail_percent=trail_percent,
            )
        target: int | None = None
        if take_profit is not None:
            target = self._exit(
                symbol,
                -quantity,
                OrderType.LIMIT,
                entry,
                exit_tif,
                f"{prefix}take_profit",
                limit_price=take_profit,
            )
        return Bracket(entry=entry, stop_loss=protect, take_profit=target)

    def _exit(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType,
        parent_id: int,
        tif: TimeInForce,
        tag: str,
        **prices: float | None,
    ) -> int:
        order_id = self._next_id()
        self._events.append(
            SignalEvent(
                self.timestamp,
                order_id=order_id,
                symbol=symbol,
                quantity=quantity,
                order_type=order_type,
                tif=tif,
                tag=tag,
                parent_id=parent_id,
                limit_price=prices.get("limit_price"),
                stop_price=prices.get("stop_price"),
                trail_amount=prices.get("trail_amount"),
                trail_percent=prices.get("trail_percent"),
            )
        )
        return order_id

    def cancel(self, order_id: int | None = None, *, symbol: str | None = None) -> None:
        """Cancel one order, all orders of ``symbol``, or (no arguments) all orders."""
        if symbol is not None:
            self._check_symbol(symbol)
        self._events.append(CancelEvent(self.timestamp, order_id=order_id, symbol=symbol))


class Strategy(ABC):
    """Base class for event-driven strategies.

    Subclasses implement :meth:`on_bar`, which is called once per bar *after* the bar
    closed and after any fills of that bar were booked. The strategy sees the market
    only through the read-only :class:`~backtester.data.MarketView`; intents placed via
    the context execute on later bars.
    """

    name: ClassVar[str] = "strategy"

    def on_start(self, ctx: StrategyContext) -> None:  # noqa: B027 - optional hook
        """Called once before the first bar."""

    @abstractmethod
    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        """React to a closed bar."""

    def on_fill(self, fill: FillEvent, ctx: StrategyContext) -> None:  # noqa: B027
        """Called for every fill of this strategy's orders."""

    def params(self) -> dict[str, Any]:
        """Parameters that identify this strategy instance (used in reports)."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


class VectorizedStrategy(Strategy):
    """A strategy whose decisions are a pure function of past closes.

    Implementing :meth:`target_weights_frame` enables the vectorized fast path
    (:func:`backtester.research.run_vectorized`). Row ``t`` of the frame must be the
    target weights decided at the close of bar ``t`` using data up to ``t`` only; the
    fast path executes them at the open of ``t + 1``, like the event engine.
    """

    @abstractmethod
    def target_weights_frame(self, feed: DataFeed) -> pd.DataFrame:
        """Target weights decided at each bar close (index = calendar, columns = symbols)."""
