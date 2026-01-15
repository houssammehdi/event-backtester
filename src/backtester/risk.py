"""Pre-trade risk checks, target-to-order translation and the drawdown kill-switch."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from backtester.errors import ConfigError
from backtester.events import CancelEvent, Event, OrderEvent, SignalEvent, TargetEvent
from backtester.orders import OrderType, Side, TimeInForce
from backtester.portfolio.portfolio import Portfolio
from backtester.portfolio.sizing import rebalance_quantities, round_to_lot, scale_to_gross


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Portfolio-level limits enforced by :class:`RiskManager`.

    Attributes:
        max_position_weight: Maximum absolute weight of any single position.
        max_gross_leverage: Maximum ``sum(|position value|) / equity``.
        max_drawdown: Drawdown from the running equity peak (e.g. ``0.25``) that
            triggers the kill-switch: all orders are cancelled, every position is
            flattened with market orders and the strategy is halted for good.
        allow_short: Whether net short positions are permitted.
    """

    max_position_weight: float | None = None
    max_gross_leverage: float | None = None
    max_drawdown: float | None = None
    allow_short: bool = True

    def __post_init__(self) -> None:
        for name in ("max_position_weight", "max_gross_leverage"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ConfigError(f"{name} must be positive")
        if self.max_drawdown is not None and not 0 < self.max_drawdown < 1:
            raise ConfigError("max_drawdown must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class RiskAction:
    """One entry of the risk manager's audit log."""

    timestamp: pd.Timestamp
    kind: str
    symbol: str
    detail: str


class RiskManager:
    """Reviews strategy intents and turns them into orders.

    * :class:`~backtester.events.TargetEvent` weights are clipped to
      ``max_position_weight``, scaled to ``max_gross_leverage`` and translated into
      market orders sized on the latest close and current equity.
    * :class:`~backtester.events.SignalEvent` order requests are reduced (or rejected)
      when the projected position - including already working orders - would breach a
      limit. Requests that reduce exposure are always allowed.
    * The drawdown kill-switch is evaluated on every bar close.

    Args:
        limits: The limits to enforce (default: none).
        lot_size: Order quantities are rounded down to multiples of this; ``None``
            allows fractional quantities.
        rebalance_threshold: Skip target rebalances when the weight deviation of a
            symbol is within this band.
        min_trade_notional: Skip target trades smaller than this value.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        *,
        lot_size: float | None = 1.0,
        rebalance_threshold: float = 0.0,
        min_trade_notional: float = 0.0,
    ) -> None:
        if rebalance_threshold < 0 or min_trade_notional < 0:
            raise ConfigError("rebalance_threshold and min_trade_notional must be >= 0")
        self.limits = limits or RiskLimits()
        self.lot_size = lot_size
        self.rebalance_threshold = rebalance_threshold
        self.min_trade_notional = min_trade_notional
        self.halted = False
        self.halted_at: pd.Timestamp | None = None
        self.peak_equity = -math.inf
        self.log: list[RiskAction] = []

    def _note(self, ts: pd.Timestamp, kind: str, symbol: str, detail: str) -> None:
        self.log.append(RiskAction(ts, kind, symbol, detail))

    # ------------------------------------------------------------------ kill switch
    def check_drawdown(self, timestamp: pd.Timestamp, equity: float) -> bool:
        """Update the equity peak; return ``True`` if the kill-switch fires *now*."""
        self.peak_equity = max(self.peak_equity, equity)
        limit = self.limits.max_drawdown
        if self.halted or limit is None or self.peak_equity <= 0:
            return False
        drawdown = 1.0 - equity / self.peak_equity
        if drawdown >= limit:
            self.halted = True
            self.halted_at = timestamp
            self._note(timestamp, "kill_switch", "*", f"drawdown {drawdown:.2%} >= {limit:.2%}")
            return True
        return False

    def flatten(
        self,
        timestamp: pd.Timestamp,
        portfolio: Portfolio,
        next_id: Callable[[], int],
    ) -> list[Event]:
        """Events that cancel all working orders and close every position (GTC market)."""
        events: list[Event] = [CancelEvent(timestamp)]
        for symbol, pos in sorted(portfolio.positions.items()):
            if pos.is_flat:
                continue
            events.append(
                OrderEvent(
                    timestamp,
                    order_id=next_id(),
                    symbol=symbol,
                    side=Side.SELL if pos.quantity > 0 else Side.BUY,
                    quantity=abs(pos.quantity),
                    order_type=OrderType.MARKET,
                    tif=TimeInForce.GTC,
                    tag="kill_switch",
                )
            )
        return events

    # ------------------------------------------------------------------ targets
    def constrain_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        """Apply shorting, per-position and gross-leverage limits to target weights."""
        lim = self.limits
        out: dict[str, float] = {}
        for symbol, w in weights.items():
            value = float(w)
            if not math.isfinite(value):
                value = 0.0
            if not lim.allow_short:
                value = max(value, 0.0)
            if lim.max_position_weight is not None:
                cap = lim.max_position_weight
                value = min(max(value, -cap), cap)
            out[symbol] = value
        if lim.max_gross_leverage is not None:
            out = scale_to_gross(out, lim.max_gross_leverage)
        return out

    def process_target(
        self,
        event: TargetEvent,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        next_id: Callable[[], int],
    ) -> list[Event]:
        """Translate a target into cancel events for stale orders plus market orders."""
        ts = event.timestamp
        if self.halted:
            self._note(ts, "rejected", "*", "target ignored: kill-switch active")
            return []
        weights = self.constrain_weights(event.weights)
        if weights != {s: float(w) for s, w in event.weights.items()}:
            self._note(ts, "clipped", "*", "target weights constrained by risk limits")
        events: list[Event] = []
        if event.partial:
            events.extend(CancelEvent(ts, symbol=s) for s in sorted(weights))
            current = {s: portfolio.quantity(s) for s in weights}
        else:
            events.append(CancelEvent(ts))
            current = {s: p.quantity for s, p in portfolio.positions.items()}
        trades = rebalance_quantities(
            weights,
            current,
            prices,
            portfolio.equity,
            lot_size=self.lot_size,
            threshold=self.rebalance_threshold,
            min_notional=self.min_trade_notional,
        )
        for symbol, qty in trades.items():
            events.append(
                OrderEvent(
                    ts,
                    order_id=next_id(),
                    symbol=symbol,
                    side=Side.from_quantity(qty),
                    quantity=abs(qty),
                    tag="rebalance",
                )
            )
        return events

    # ------------------------------------------------------------------ explicit orders
    def review_signal(
        self,
        event: SignalEvent,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        pending: Mapping[str, float],
    ) -> tuple[OrderEvent, bool]:
        """Review an explicit order request.

        Returns:
            ``(order_event, approved)``. When not approved the event still describes the
            original request so it can be logged as rejected.
        """
        ts, symbol = event.timestamp, event.symbol
        original = OrderEvent(
            ts,
            order_id=event.order_id,
            symbol=symbol,
            side=Side.from_quantity(event.quantity),
            quantity=abs(event.quantity),
            order_type=event.order_type,
            tif=event.tif,
            limit_price=event.limit_price,
            stop_price=event.stop_price,
            tag=event.tag,
        )
        if self.halted:
            self._note(ts, "rejected", symbol, "order rejected: kill-switch active")
            return original, False
        qty = self._allowed_quantity(event, portfolio, prices, pending)
        if qty == 0:
            self._note(ts, "rejected", symbol, "order would breach risk limits")
            return original, False
        if qty != event.quantity:
            self._note(ts, "clipped", symbol, f"quantity {event.quantity:g} -> {qty:g}")
        approved = OrderEvent(
            ts,
            order_id=event.order_id,
            symbol=symbol,
            side=Side.from_quantity(qty),
            quantity=abs(qty),
            order_type=event.order_type,
            tif=event.tif,
            limit_price=event.limit_price,
            stop_price=event.stop_price,
            tag=event.tag,
        )
        return approved, True

    def _allowed_quantity(
        self,
        event: SignalEvent,
        portfolio: Portfolio,
        prices: Mapping[str, float],
        pending: Mapping[str, float],
    ) -> float:
        lim = self.limits
        symbol, qty = event.symbol, event.quantity
        price = prices.get(symbol, math.nan)
        if event.limit_price is not None:
            price = event.limit_price
        elif event.stop_price is not None:
            price = event.stop_price
        before = portfolio.quantity(symbol) + pending.get(symbol, 0.0)
        after = before + qty
        if not lim.allow_short and qty < 0:
            after = max(after, min(before, 0.0))
        reduces = abs(after) <= abs(before) and after * before >= 0
        equity = portfolio.equity
        if not reduces and math.isfinite(price) and price > 0 and equity > 0:
            cap = math.inf
            if lim.max_position_weight is not None:
                cap = min(cap, lim.max_position_weight * equity / price)
            if lim.max_gross_leverage is not None:
                own = abs(portfolio.position(symbol).market_value)
                other_gross = portfolio.gross_exposure - own
                room = lim.max_gross_leverage * equity - max(other_gross, 0.0)
                cap = min(cap, max(room, 0.0) / price)
            if abs(after) > cap:
                after = math.copysign(cap, after)
        allowed = after - before
        if allowed * qty <= 0:
            return 0.0
        return round_to_lot(allowed, self.lot_size) if allowed != qty else qty
