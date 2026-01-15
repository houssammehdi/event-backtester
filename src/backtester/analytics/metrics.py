"""Performance metrics.

All functions are pure and operate on plain NumPy/pandas inputs so they can be tested
against hand-computed values. Conventions:

* ``returns`` are simple per-period returns; annualisation uses ``periods_per_year``.
* Volatility-based ratios use the sample standard deviation (``ddof=1``).
* Drawdowns are reported as positive fractions (``0.25`` means -25 %).
* Undefined ratios (zero variance, no losing trades, ...) are ``nan`` or ``inf`` rather
  than silently zero.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from backtester.analytics.benchmark import BenchmarkStats
    from backtester.engine import BacktestResult

ArrayLike = npt.ArrayLike


def _arr(values: ArrayLike) -> npt.NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64)
    return out[~np.isnan(out)]


def total_return(equity: ArrayLike) -> float:
    """``equity[-1] / equity[0] - 1``."""
    e = _arr(equity)
    if len(e) < 1 or e[0] == 0:
        return math.nan
    return float(e[-1] / e[0] - 1.0)


def cagr(equity: ArrayLike, periods_per_year: float = 252.0) -> float:
    """Compound annual growth rate, with ``years = (len(equity) - 1) / periods_per_year``."""
    e = _arr(equity)
    if len(e) < 2 or e[0] <= 0:
        return math.nan
    years = (len(e) - 1) / periods_per_year
    if e[-1] <= 0:
        return -1.0
    return float((e[-1] / e[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: ArrayLike, periods_per_year: float = 252.0) -> float:
    """Sample standard deviation of returns times ``sqrt(periods_per_year)``."""
    r = _arr(returns)
    if len(r) < 2:
        return math.nan
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: float = 252.0, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sharpe ratio ``mean(r - rf) / std(r) * sqrt(periods_per_year)``.

    ``risk_free_rate`` is annual and converted to a per-period rate by division.
    """
    r = _arr(returns) - risk_free_rate / periods_per_year
    if len(r) < 2:
        return math.nan
    sd = float(np.std(r, ddof=1))
    if sd == 0:
        return math.nan
    return float(np.mean(r) / sd * math.sqrt(periods_per_year))


def downside_deviation(returns: ArrayLike, target: float = 0.0) -> float:
    """Root-mean-square of shortfalls below ``target`` over *all* observations."""
    r = _arr(returns)
    if len(r) == 0:
        return math.nan
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)))


def sortino_ratio(
    returns: ArrayLike, periods_per_year: float = 252.0, risk_free_rate: float = 0.0
) -> float:
    """Annualised Sortino ratio ``mean(r - rf) / downside_deviation * sqrt(ppy)``."""
    rf = risk_free_rate / periods_per_year
    r = _arr(returns)
    if len(r) < 2:
        return math.nan
    dd = downside_deviation(r, target=rf)
    mean_excess = float(np.mean(r - rf))
    if dd == 0:
        return math.inf if mean_excess > 0 else math.nan
    return mean_excess / dd * math.sqrt(periods_per_year)


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Drawdown from the running peak at every bar, as a positive fraction."""
    peak = equity.cummax()
    return (1.0 - equity / peak).rename("drawdown")


def max_drawdown(equity: ArrayLike) -> float:
    """Largest peak-to-trough decline as a positive fraction."""
    e = _arr(equity)
    if len(e) == 0:
        return math.nan
    peak = np.maximum.accumulate(e)
    return float(np.max(1.0 - e / peak))


def max_drawdown_duration(equity: ArrayLike) -> int:
    """Longest stretch, in bars, spent below a previous equity peak.

    A drawdown lasts from the bar after a peak until the bar that makes a new high
    (or the end of the series if it never recovers).
    """
    e = _arr(equity)
    longest = current = 0
    peak = -math.inf
    for value in e:
        if value >= peak:
            peak = value
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def calmar_ratio(cagr_value: float, max_dd: float) -> float:
    """``CAGR / max drawdown`` (``nan`` when there was no drawdown)."""
    if not max_dd or math.isnan(max_dd):
        return math.nan
    return cagr_value / max_dd


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Statistics over closed round-trip trades (net of commissions)."""

    n_trades: int
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    avg_bars_held: float


def trade_stats(trades: pd.DataFrame) -> TradeStats:
    """Compute :class:`TradeStats` from a :func:`~backtester.analytics.round_trips` log.

    * hit rate - share of closed trades with positive net PnL;
    * profit factor - gross profits of winners over gross losses of losers;
    * avg win / avg loss - mean net PnL of winners / losers (loss is negative).
    """
    closed = trades.loc[trades["status"] == "closed"] if len(trades) else trades
    n = len(closed)
    if n == 0:
        return TradeStats(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    pnl = closed["net_pnl"].to_numpy(dtype=np.float64)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = -float(losses.sum())
    if gross_loss > 0:
        profit_factor = float(wins.sum()) / gross_loss
    else:
        profit_factor = math.inf if len(wins) else math.nan
    return TradeStats(
        n_trades=n,
        hit_rate=len(wins) / n,
        profit_factor=profit_factor,
        avg_win=float(wins.mean()) if len(wins) else math.nan,
        avg_loss=float(losses.mean()) if len(losses) else math.nan,
        avg_trade=float(pnl.mean()),
        avg_bars_held=float(closed["bars_held"].mean()),
    )


def annualized_turnover(
    traded_notional: ArrayLike, equity: ArrayLike, periods_per_year: float = 252.0
) -> float:
    """Two-sided traded notional per year divided by average equity.

    Buying a full portfolio once and never trading again over one year is 100 %.
    """
    t = np.asarray(traded_notional, dtype=np.float64)
    e = np.asarray(equity, dtype=np.float64)
    if len(e) < 2:
        return math.nan
    years = (len(e) - 1) / periods_per_year
    return float(t.sum() / np.mean(e) / years)


def average_exposure(gross: ArrayLike, equity: ArrayLike) -> float:
    """Mean of ``gross exposure / equity`` over all bars."""
    g = np.asarray(gross, dtype=np.float64)
    e = np.asarray(equity, dtype=np.float64)
    return float(np.mean(np.where(e > 0, g / e, np.nan)))


def time_in_market(gross: ArrayLike) -> float:
    """Fraction of bars with a non-zero position."""
    g = np.asarray(gross, dtype=np.float64)
    return float(np.mean(g > 0)) if len(g) else math.nan


@dataclass(frozen=True, slots=True)
class Metrics:
    """Summary statistics of a backtest."""

    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int
    initial_equity: float
    final_equity: float
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar: float
    trades: TradeStats
    turnover: float
    exposure: float
    time_in_market: float
    total_commission: float
    total_slippage: float
    total_borrow_cost: float
    benchmark: BenchmarkStats | None

    def as_dict(self) -> dict[str, object]:
        """Flat dictionary (trade and benchmark stats inlined with prefixes)."""
        out: dict[str, object] = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                prefix = "trade_" if key == "trades" else "benchmark_"
                out.update({f"{prefix}{k}": v for k, v in value.items()})
            elif value is not None:
                out[key] = value
        return out


def compute_metrics(result: BacktestResult, risk_free_rate: float = 0.0) -> Metrics:
    """Compute :class:`Metrics` for a backtest result, including a benchmark comparison."""
    from backtester.analytics.benchmark import compare

    ppy = result.periods_per_year
    eq = result.equity
    r = result.returns
    cagr_value = cagr(eq, ppy)
    mdd = max_drawdown(eq)
    bench = result.benchmark_equity()
    bench_stats = compare(r, bench.pct_change().iloc[1:], bench, ppy) if len(eq) > 2 else None
    return Metrics(
        start=eq.index[0],
        end=eq.index[-1],
        n_bars=len(eq),
        initial_equity=float(eq.iloc[0]),
        final_equity=float(eq.iloc[-1]),
        total_return=total_return(eq),
        cagr=cagr_value,
        annual_volatility=annualized_volatility(r, ppy),
        sharpe=sharpe_ratio(r, ppy, risk_free_rate),
        sortino=sortino_ratio(r, ppy, risk_free_rate),
        max_drawdown=mdd,
        max_drawdown_duration=max_drawdown_duration(eq),
        calmar=calmar_ratio(cagr_value, mdd),
        trades=trade_stats(result.trades()),
        turnover=annualized_turnover(result.exposure["traded"], eq, ppy),
        exposure=average_exposure(result.exposure["gross"], eq),
        time_in_market=time_in_market(result.exposure["gross"]),
        total_commission=result.total_commission,
        total_slippage=result.total_slippage,
        total_borrow_cost=result.total_borrow_cost,
        benchmark=bench_stats,
    )
