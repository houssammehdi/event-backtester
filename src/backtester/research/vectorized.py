"""Vectorized fast path for target-weight strategies.

:func:`run_vectorized` reproduces the event engine's accounting for strategies that
implement :class:`~backtester.strategy.VectorizedStrategy` - without an event queue,
order objects or per-bar Python callbacks:

* weights decided at the close of bar ``t`` are sized with equity and closes of ``t``
  and executed at the open of ``t + 1`` (market orders, fractional quantities) - or,
  with ``execution="close"``, at the close of ``t + 1`` (market-on-close orders);
* between rebalances holdings are constant, so equity for the whole block is a single
  matrix product ``cash + closes @ shares``;
* costs are a fixed adverse slippage in bps of the execution price plus a commission
  in bps of traded notional - the same as ``FixedBpsSlippage`` + ``BpsCommission``.

The loop runs over rebalance dates only (monthly strategies: ~12 iterations per year).
Features that need per-order state - limit/stop orders, participation caps, risk
limits, the kill-switch, missing bars - are the event engine's job; the parity tests
check both paths agree to numerical precision where both apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from backtester.analytics.metrics import cagr, max_drawdown, sharpe_ratio, total_return
from backtester.data.feed import DataFeed
from backtester.errors import ConfigError
from backtester.strategy.base import VectorizedStrategy


@dataclass(frozen=True, slots=True)
class VectorizedResult:
    """Output of :func:`run_vectorized`."""

    equity: pd.Series
    positions: pd.DataFrame
    traded: pd.Series
    total_costs: float
    periods_per_year: float

    @property
    def returns(self) -> pd.Series:
        """Bar-to-bar returns of the equity curve."""
        return self.equity.pct_change().iloc[1:].rename("returns")

    def summary(self) -> dict[str, float]:
        """Headline statistics."""
        eq = self.equity
        return {
            "total_return": total_return(eq),
            "cagr": cagr(eq, self.periods_per_year),
            "sharpe": sharpe_ratio(self.returns, self.periods_per_year),
            "max_drawdown": max_drawdown(eq),
        }


def _position(feed: DataFeed, where: int | None, default: int) -> int:
    if where is None:
        return default
    pos = where + len(feed) if where < 0 else where
    if not 0 <= pos < len(feed):
        raise ConfigError(f"window bound {where} outside the data")
    return pos


def run_vectorized(
    feed: DataFeed,
    strategy: VectorizedStrategy,
    *,
    initial_cash: float = 1_000_000.0,
    slippage_bps: float = 0.0,
    commission_bps: float = 0.0,
    start: int | None = None,
    end: int | None = None,
    periods_per_year: float = 252.0,
    execution: Literal["open", "close"] = "open",
) -> VectorizedResult:
    """Backtest ``strategy`` with the vectorized fast path.

    Args:
        feed: Market data; every symbol must have a bar on every calendar date of the
            window (use the event engine for data with holes).
        strategy: A strategy providing :meth:`~VectorizedStrategy.target_weights_frame`.
        initial_cash: Starting capital.
        slippage_bps: Adverse execution slippage relative to the open, in bps.
        commission_bps: Commission in bps of traded notional.
        start: First calendar position of the trading window (earlier rows are warm-up).
        end: Last calendar position (inclusive).
        periods_per_year: Bars per year for annualised statistics.
        execution: Trade at the next bar's ``"open"`` (like market orders) or its
            ``"close"`` (like market-on-close orders, i.e. ``target_order_type`` set to
            ``MARKET_ON_CLOSE`` in the event engine).
    """
    if initial_cash <= 0 or slippage_bps < 0 or commission_bps < 0:
        raise ConfigError("initial_cash must be positive and costs non-negative")
    if execution not in ("open", "close"):
        raise ConfigError(f"execution must be 'open' or 'close', got {execution!r}")
    lo = _position(feed, start, 0)
    hi = _position(feed, end, len(feed) - 1)
    if lo > hi:
        raise ConfigError("start must not be after end")
    if not feed.has_bar[lo : hi + 1].all():
        raise ConfigError("run_vectorized needs complete bars; use the event engine instead")

    weights = strategy.target_weights_frame(feed).reindex(columns=list(feed.symbols))
    w = weights.to_numpy(dtype=np.float64)
    decisions = [t for t in range(lo, hi) if not np.isnan(w[t]).all()]
    if lo < hi and (not decisions or decisions[0] != lo):
        # Establish the most recently scheduled portfolio on the first trading bar.
        prior = [t for t in range(lo) if not np.isnan(w[t]).all()]
        if prior:
            w = w.copy()
            w[lo] = w[prior[-1]]
            decisions.insert(0, lo)

    closes = feed.last_close[lo : hi + 1]
    fills = closes if execution == "close" else feed.array("open")[lo : hi + 1]
    n, m = closes.shape
    slip = slippage_bps / 10_000.0
    comm = commission_bps / 10_000.0
    equity = np.empty(n)
    held = np.zeros((n, m))
    traded = np.zeros(n)
    shares = np.zeros(m)
    cash = float(initial_cash)
    costs = 0.0
    block = 0
    for d in decisions:
        k = d - lo
        e = k + 1
        equity[block:e] = cash + closes[block:e] @ shares
        held[block:e] = shares
        target = np.nan_to_num(w[d]) * equity[k] / closes[k]
        delta = target - shares
        price = fills[e] * (1.0 + np.sign(delta) * slip)
        notional = np.abs(delta) * price
        fee = float(notional.sum()) * comm
        cash -= float(delta @ price) + fee
        costs += fee + float((np.abs(delta) * fills[e] * slip).sum())
        traded[e] = float(notional.sum())
        shares = target
        block = e
    equity[block:] = cash + closes[block:] @ shares
    held[block:] = shares

    index = feed.index[lo : hi + 1]
    return VectorizedResult(
        equity=pd.Series(equity, index=index, name="equity"),
        positions=pd.DataFrame(held, index=index, columns=feed.symbols),
        traded=pd.Series(traded, index=index, name="traded"),
        total_costs=costs,
        periods_per_year=periods_per_year,
    )
