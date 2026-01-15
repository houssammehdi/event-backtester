"""Round-trip trade reconstruction from a fill log."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "symbol",
    "direction",
    "status",
    "entry_time",
    "exit_time",
    "quantity",
    "entry_price",
    "exit_price",
    "pnl",
    "commission",
    "net_pnl",
    "return",
    "bars_held",
]


@dataclass(slots=True)
class _Trip:
    symbol: str
    direction: int
    entry_time: pd.Timestamp
    position: float = 0.0
    max_quantity: float = 0.0
    entry_qty: float = 0.0
    entry_notional: float = 0.0
    exit_qty: float = 0.0
    exit_notional: float = 0.0
    avg_cost: float = 0.0
    pnl: float = 0.0
    commission: float = 0.0
    exit_time: pd.Timestamp | None = field(default=None)

    def add(self, qty: float, price: float, commission: float) -> None:
        total = abs(self.position) + qty
        self.avg_cost = (abs(self.position) * self.avg_cost + qty * price) / total
        self.position += self.direction * qty
        self.max_quantity = max(self.max_quantity, abs(self.position))
        self.entry_qty += qty
        self.entry_notional += qty * price
        self.commission += commission

    def reduce(self, qty: float, price: float, commission: float) -> None:
        self.pnl += (price - self.avg_cost) * qty * self.direction
        self.position -= self.direction * qty
        self.exit_qty += qty
        self.exit_notional += qty * price
        self.commission += commission

    def row(self, index: pd.DatetimeIndex, last_price: float) -> dict[str, object]:
        is_open = self.exit_time is None
        pnl = self.pnl
        if is_open and math.isfinite(last_price):
            pnl += (last_price - self.avg_cost) * abs(self.position) * self.direction
        entry_price = self.entry_notional / self.entry_qty
        exit_price = self.exit_notional / self.exit_qty if self.exit_qty else math.nan
        net = pnl - self.commission
        end = index[-1] if self.exit_time is None else self.exit_time
        bars = int(index.searchsorted(end)) - int(index.searchsorted(self.entry_time))
        return {
            "symbol": self.symbol,
            "direction": "long" if self.direction > 0 else "short",
            "status": "open" if is_open else "closed",
            "entry_time": self.entry_time,
            "exit_time": pd.NaT if is_open else self.exit_time,
            "quantity": self.max_quantity,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "commission": self.commission,
            "net_pnl": net,
            "return": net / self.entry_notional if self.entry_notional else math.nan,
            "bars_held": bars,
        }


def round_trips(fills: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Group fills into round-trip trades.

    A round trip starts when a symbol's position leaves zero and ends when it returns
    to zero. A fill that flips the position closes the current trip and opens a new
    one with the remainder; its commission is split pro rata. Scaling in and out within
    a trip is supported (average-cost accounting).

    Trades still open at the end are included with ``status == "open"`` and PnL marked
    at the last available price; trade statistics use closed trades only.

    Args:
        fills: Fill log with columns ``timestamp, symbol, side, quantity, price,
            commission`` (as in :attr:`BacktestResult.fills`).
        prices: Price frame whose index is the calendar used to count ``bars_held`` and
            whose last row marks open trades.
    """
    rows: list[dict[str, object]] = []
    if len(fills) == 0:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in TRADE_COLUMNS})
    index = pd.DatetimeIndex(prices.index)
    ordered = fills.sort_values(["timestamp", "order_id"], kind="stable")
    for symbol, group in ordered.groupby("symbol", sort=True):
        sym = str(symbol)
        trip: _Trip | None = None
        last = prices[sym].iloc[-1] if sym in prices.columns else math.nan
        for ts, side, qty, price, comm in zip(
            group["timestamp"],
            group["side"],
            group["quantity"].to_numpy(dtype=np.float64),
            group["price"].to_numpy(dtype=np.float64),
            group["commission"].to_numpy(dtype=np.float64),
            strict=True,
        ):
            sign = 1 if side == "buy" else -1
            remaining = float(qty)
            while remaining > 1e-12:
                if trip is None:
                    trip = _Trip(sym, sign, pd.Timestamp(ts))
                if sign == trip.direction:
                    trip.add(remaining, float(price), float(comm) * remaining / qty)
                    remaining = 0.0
                    continue
                closed = min(remaining, abs(trip.position))
                trip.reduce(closed, float(price), float(comm) * closed / qty)
                remaining -= closed
                if abs(trip.position) <= 1e-9:
                    trip.exit_time = pd.Timestamp(ts)
                    rows.append(trip.row(index, float(last)))
                    trip = None
        if trip is not None:
            rows.append(trip.row(index, float(last)))
    out = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    return out.sort_values(["entry_time", "symbol"], kind="stable").reset_index(drop=True)
