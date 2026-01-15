"""Strategy interface and the context object strategies use to act."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, ClassVar

import pandas as pd

from backtester.data.feed import DataFeed, MarketView
from backtester.events import CancelEvent, Event, FillEvent, SignalEvent, TargetEvent
from backtester.execution.broker import SimulatedBroker
from backtester.orders import Order, OrderType, TimeInForce
from backtester.portfolio.portfolio import Portfolio


class StrategyContext:
    """The strategy's handle on its account and on the order flow.

    Read methods expose account state (cash, equity, positions, working orders).
    Action methods do not act immediately: they record intents - a
    :class:`~backtester.events.TargetEvent`, :class:`~backtester.events.SignalEvent` or
    :class:`~backtester.events.CancelEvent` - that the engine pushes through the event
    queue to the risk manager and broker after ``on_bar`` returns.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        broker: SimulatedBroker,
        next_id: Callable[[], int] | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._broker = broker
        self._next_id = next_id or itertools.count(1).__next__
        self._events: list[Event] = []
        self._timestamp: pd.Timestamp | None = None
        self._muted = False

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
        return [replace(o) for o in self._broker.open_orders(symbol)]

    # ------------------------------------------------------------------ actions
    def target_weights(self, weights: Mapping[str, float], *, partial: bool = False) -> None:
        """Request a rebalance to ``weights`` (fractions of equity) at the next open.

        With ``partial=False`` symbols not listed are closed and every working order is
        cancelled; with ``partial=True`` only the listed symbols are touched.
        """
        self._events.append(TargetEvent(self.timestamp, weights=weights, partial=partial))

    def order(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
        tag: str = "",
    ) -> int:
        """Request an order for a signed ``quantity``; returns the order id."""
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
            )
        )
        return order_id

    def cancel(self, order_id: int | None = None, *, symbol: str | None = None) -> None:
        """Cancel one order, all orders of ``symbol``, or (no arguments) all orders."""
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
