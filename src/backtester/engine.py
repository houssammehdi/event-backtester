"""The event loop wiring DataFeed -> Strategy -> RiskManager -> Broker -> Portfolio."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from backtester.data.feed import DataFeed, MarketView
from backtester.errors import AccountingError, ConfigError
from backtester.events import (
    CancelEvent,
    Event,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    TargetEvent,
)
from backtester.execution.broker import SimulatedBroker
from backtester.orders import Order
from backtester.portfolio.portfolio import Portfolio
from backtester.risk import RiskAction, RiskLimits, RiskManager
from backtester.strategy.base import Strategy, StrategyContext

if TYPE_CHECKING:
    from pathlib import Path

    from backtester.analytics.metrics import Metrics

_FILL_COLUMNS = [
    "timestamp",
    "order_id",
    "symbol",
    "side",
    "quantity",
    "price",
    "commission",
    "slippage",
]


@dataclass(slots=True)
class BacktestResult:
    """Everything produced by one backtest run.

    Time series are indexed by the bars of the trading window (bar closes).
    """

    strategy_name: str
    params: dict[str, Any]
    initial_cash: float
    periods_per_year: float
    equity: pd.Series
    cash: pd.Series
    exposure: pd.DataFrame
    """Columns ``gross``, ``net`` (currency) and ``traded`` (traded notional per bar)."""
    positions: pd.DataFrame
    """Signed quantities per symbol at each bar close."""
    weights: pd.DataFrame
    """Signed position weights (value / equity) at each bar close."""
    prices: pd.DataFrame
    """Forward-filled closes over the trading window (for benchmarks)."""
    fills: pd.DataFrame
    orders: list[Order]
    risk_log: list[RiskAction]
    halted_at: pd.Timestamp | None = None
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_borrow_cost: float = 0.0
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def returns(self) -> pd.Series:
        """Simple bar-to-bar returns of the equity curve (one fewer than bars)."""
        return self.equity.pct_change().iloc[1:].rename("returns")

    @property
    def final_equity(self) -> float:
        """Equity at the last bar."""
        return float(self.equity.iloc[-1])

    def trades(self) -> pd.DataFrame:
        """Round-trip trade log reconstructed from the fills."""
        if "trades" not in self._cache:
            from backtester.analytics.trades import round_trips

            self._cache["trades"] = round_trips(self.fills, self.prices)
        trades: pd.DataFrame = self._cache["trades"]
        return trades

    def benchmark_equity(self) -> pd.Series:
        """Equal-weight buy-and-hold of the universe, scaled to ``initial_cash``."""
        from backtester.analytics.benchmark import buy_and_hold

        return buy_and_hold(self.prices, self.initial_cash)

    def metrics(self, risk_free_rate: float = 0.0) -> Metrics:
        """Performance metrics of the run."""
        from backtester.analytics.metrics import compute_metrics

        return compute_metrics(self, risk_free_rate=risk_free_rate)

    def report(self, risk_free_rate: float = 0.0) -> str:
        """Human-readable performance report."""
        from backtester.analytics.report import format_report

        return format_report(self, risk_free_rate=risk_free_rate)

    def plot(self, path: str | Path | None = None) -> Any:
        """Plot equity and drawdown (requires the ``plot`` extra)."""
        from backtester.plotting import plot_result

        return plot_result(self, path=path)


class Engine:
    """Runs one strategy over one data feed.

    Processing of each bar ``t``:

    1. The broker matches orders working from earlier bars against bar ``t`` and emits
       :class:`~backtester.events.FillEvent` s.
    2. A :class:`~backtester.events.MarketEvent` for ``t`` is queued behind them.
    3. Events are drained in ``(timestamp, priority, sequence)`` order: fills update the
       portfolio; the market event marks the book to the close, records equity, runs the
       drawdown kill-switch and calls the strategy; strategy intents go to the risk
       manager, which emits orders for the broker. Those orders can first execute on
       bar ``t + 1``.

    Args:
        feed: Market data.
        strategy: The strategy instance (stateful; use a fresh one per run).
        initial_cash: Starting capital.
        broker: Execution simulator (default: frictionless, no participation cap).
        risk: A :class:`~backtester.risk.RiskManager`, or :class:`~backtester.risk.RiskLimits`
            to build one with default sizing (whole shares).
        periods_per_year: Bars per year, used for borrow accrual and annualisation.
        borrow_rate: Annual fee charged on the value of short positions.
        start: First calendar position (or timestamp) of the trading window. Earlier
            bars are replayed to the strategy as warm-up, but its intents are dropped
            and no equity is recorded.
        end: Last calendar position (or timestamp) of the trading window, inclusive.
        check_invariants: Verify the accounting identities on every bar.
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        *,
        initial_cash: float = 1_000_000.0,
        broker: SimulatedBroker | None = None,
        risk: RiskManager | RiskLimits | None = None,
        periods_per_year: float = 252.0,
        borrow_rate: float = 0.0,
        start: int | pd.Timestamp | str | None = None,
        end: int | pd.Timestamp | str | None = None,
        check_invariants: bool = False,
    ) -> None:
        if periods_per_year <= 0:
            raise ConfigError("periods_per_year must be positive")
        if borrow_rate < 0:
            raise ConfigError("borrow_rate must be non-negative")
        self.feed = feed
        self.strategy = strategy
        self.broker = broker or SimulatedBroker()
        if isinstance(risk, RiskManager):
            self.risk = risk
        else:
            self.risk = RiskManager(risk)
        self.portfolio = Portfolio(initial_cash, feed.symbols)
        self.periods_per_year = periods_per_year
        self.borrow_rate = borrow_rate
        self.check_invariants = check_invariants
        self._start = self._resolve(start, 0, is_end=False)
        self._end = self._resolve(end, len(feed) - 1, is_end=True)
        if self._start > self._end:
            raise ConfigError("start must not be after end")
        self._ids = itertools.count(1)
        self.context = StrategyContext(
            self.portfolio, self.broker, self._next_id, symbols=feed.symbols
        )
        self._queue = EventQueue()
        self._view = feed.view(0)
        self._ran = False

    def _next_id(self) -> int:
        return next(self._ids)

    def _resolve(
        self, where: int | pd.Timestamp | str | None, default: int, *, is_end: bool
    ) -> int:
        """Map a window bound to a calendar position.

        Integers are positions (negative counts from the end). Timestamps resolve to the
        first bar at or after them for ``start`` and the last bar at or before them for
        ``end``.
        """
        if where is None:
            return default
        if isinstance(where, int | np.integer):
            pos = int(where)
            if pos < 0:
                pos += len(self.feed)
        else:
            ts = pd.Timestamp(where)
            if is_end:
                pos = int(self.feed.index.searchsorted(ts, side="right")) - 1
            else:
                pos = int(self.feed.index.searchsorted(ts, side="left"))
        if not 0 <= pos < len(self.feed):
            raise ConfigError(f"window bound {where!r} is outside the data")
        return pos

    # ------------------------------------------------------------------ main loop
    def run(self) -> BacktestResult:
        """Run the backtest and return its :class:`BacktestResult`."""
        if self._ran:
            raise RuntimeError("an Engine instance can only run once; create a new one")
        self._ran = True
        n_rec = self._end - self._start + 1
        symbols = self.feed.symbols
        equity = np.empty(n_rec)
        cash = np.empty(n_rec)
        gross = np.empty(n_rec)
        net = np.empty(n_rec)
        traded = np.zeros(n_rec)
        positions = np.zeros((n_rec, len(symbols)))
        values = np.zeros((n_rec, len(symbols)))
        fills: list[FillEvent] = []

        self.strategy.on_start(self.context)
        for i in range(self._end + 1):
            view = self._view = self.feed.view(i)
            ts = view.timestamp
            if i >= self._start:
                for fill in self.broker.process_bar(view):
                    self._queue.push(fill)
            self._queue.push(MarketEvent(ts, index=i))
            while self._queue:
                event = self._queue.pop()
                if isinstance(event, FillEvent):
                    self.portfolio.on_fill(event)
                    fills.append(event)
                    traded[i - self._start] += event.notional
                    if not self.risk.halted:
                        # A halted strategy is never called again, not even for the
                        # kill switch's own fills: its intents could cancel the
                        # flattening orders.
                        self.context._begin(ts)
                        self.strategy.on_fill(event, self.context)
                        self._push_all(self.context._drain())
                elif isinstance(event, MarketEvent):
                    self._on_market(view)
                    if i >= self._start:
                        k = i - self._start
                        equity[k] = self.portfolio.equity
                        cash[k] = self.portfolio.cash
                        gross[k] = self.portfolio.gross_exposure
                        net[k] = self.portfolio.net_exposure
                        for j, s in enumerate(symbols):
                            pos = self.portfolio.positions[s]
                            positions[k, j] = pos.quantity
                            values[k, j] = pos.market_value
                else:
                    self._dispatch(event)

        index = self.feed.index[self._start : self._end + 1]
        eq = pd.Series(equity, index=index, name="equity")
        with np.errstate(divide="ignore", invalid="ignore"):
            weight_values = np.where(equity[:, None] > 0, values / equity[:, None], np.nan)
        prices = pd.DataFrame(
            self.feed.last_close[self._start : self._end + 1].copy(), index=index, columns=symbols
        )
        return BacktestResult(
            strategy_name=type(self.strategy).name,
            params=self.strategy.params(),
            initial_cash=self.portfolio.initial_cash,
            periods_per_year=self.periods_per_year,
            equity=eq,
            cash=pd.Series(cash, index=index, name="cash"),
            exposure=pd.DataFrame({"gross": gross, "net": net, "traded": traded}, index=index),
            positions=pd.DataFrame(positions, index=index, columns=symbols),
            weights=pd.DataFrame(weight_values, index=index, columns=symbols),
            prices=prices,
            fills=_fills_frame(fills),
            orders=self.broker.orders,
            risk_log=list(self.risk.log),
            halted_at=self.risk.halted_at,
            total_commission=self.portfolio.total_commission,
            total_slippage=self.portfolio.total_slippage,
            total_borrow_cost=self.portfolio.total_borrow_cost,
        )

    # ------------------------------------------------------------------ handlers
    def _push_all(self, events: list[Event]) -> None:
        for event in events:
            self._queue.push(event)

    def _on_market(self, view: MarketView) -> None:
        ts = view.timestamp
        trading = view.position >= self._start
        prices = view.prices()
        self.portfolio.mark(prices)
        if trading:
            self.portfolio.charge_borrow(self.borrow_rate, 1.0 / self.periods_per_year)
            if self.check_invariants:
                self._verify(ts, prices)
            if self.risk.check_drawdown(ts, self.portfolio.equity):
                self._push_all(self.risk.flatten(ts, self.portfolio, self._next_id))
        if self.risk.halted:
            return
        self.context._begin(ts, muted=not trading)
        self.strategy.on_bar(view, self.context)
        self._push_all(self.context._drain())

    def _dispatch(self, event: Event) -> None:
        if isinstance(event, TargetEvent):
            self._push_all(
                self.risk.process_target(event, self.portfolio, self._view.prices(), self._next_id)
            )
        elif isinstance(event, SignalEvent):
            pending: dict[str, float] = {}
            for order in self.broker.open_orders(event.symbol):
                pending[order.symbol] = pending.get(order.symbol, 0.0) + order.signed_remaining
            order_event, approved = self.risk.review_signal(
                event, self.portfolio, self._view.prices(), pending
            )
            if approved:
                self._queue.push(order_event)
            else:
                self.broker.record_rejection(order_event)
        elif isinstance(event, CancelEvent):
            if event.order_id is not None:
                self.broker.cancel(event.order_id, event.timestamp)
            else:
                self.broker.cancel_all(event.timestamp, event.symbol)
        elif isinstance(event, OrderEvent):
            self.broker.submit(event)
        else:  # pragma: no cover - defensive
            raise TypeError(f"unhandled event {event!r}")

    def _verify(self, ts: pd.Timestamp, prices: Mapping[str, float]) -> None:
        """Check both accounting identities against independently sourced prices."""
        p = self.portfolio
        direct = p.cash + sum(
            pos.quantity * prices[s] for s, pos in p.positions.items() if not pos.is_flat
        )
        tolerance = 1e-7 * max(abs(p.initial_cash), 1.0)
        if not math.isclose(direct, p.equity, abs_tol=tolerance):
            raise AccountingError(f"{ts}: equity != cash + positions ({direct} vs {p.equity})")
        if p.identity_error() > tolerance:
            error = p.identity_error()
            raise AccountingError(f"{ts}: PnL attribution identity violated by {error}")


def _fills_frame(fills: list[FillEvent]) -> pd.DataFrame:
    if not fills:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in _FILL_COLUMNS}).astype(
            {"timestamp": "datetime64[ns]", "symbol": "object", "side": "object"}
        )
    return pd.DataFrame(
        {
            "timestamp": [f.timestamp for f in fills],
            "order_id": [f.order_id for f in fills],
            "symbol": [f.symbol for f in fills],
            "side": [f.side.name.lower() for f in fills],
            "quantity": [f.quantity for f in fills],
            "price": [f.price for f in fills],
            "commission": [f.commission for f in fills],
            "slippage": [f.slippage for f in fills],
        }
    )
