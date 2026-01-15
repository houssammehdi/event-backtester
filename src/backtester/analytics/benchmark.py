"""Benchmark construction and relative performance (beta, alpha, information ratio)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.analytics.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


def buy_and_hold(prices: pd.DataFrame, initial_cash: float = 1.0) -> pd.Series:
    """Equal-weight buy-and-hold equity curve, bought at the first row's prices.

    Symbols without a price in the first row are excluded; there is no rebalancing,
    so weights drift with performance.
    """
    first = prices.iloc[0]
    held = first.index[first.notna() & (first > 0)]
    if len(held) == 0:
        return pd.Series(initial_cash, index=prices.index, name="benchmark")
    rel = prices[held].ffill() / first[held]
    return (rel.mean(axis=1) * initial_cash).rename("benchmark")


@dataclass(frozen=True, slots=True)
class BenchmarkStats:
    """Performance of a strategy relative to a benchmark."""

    beta: float
    alpha: float
    """Annualised Jensen alpha with a zero risk-free rate."""
    correlation: float
    tracking_error: float
    information_ratio: float
    total_return: float
    cagr: float
    sharpe: float
    volatility: float
    max_drawdown: float


def beta_alpha(
    returns: npt.ArrayLike,
    benchmark_returns: npt.ArrayLike,
    periods_per_year: float = 252.0,
) -> tuple[float, float]:
    """OLS beta and annualised Jensen alpha (zero risk-free rate).

    ``beta = cov(r, rb) / var(rb)`` and ``alpha = (mean(r) - beta * mean(rb)) * ppy``.
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    mask = ~(np.isnan(r) | np.isnan(b))
    r, b = r[mask], b[mask]
    if len(r) < 2:
        return math.nan, math.nan
    var_b = float(np.var(b, ddof=1))
    if var_b == 0:
        return math.nan, math.nan
    beta = float(np.cov(r, b, ddof=1)[0, 1] / var_b)
    alpha = (float(np.mean(r)) - beta * float(np.mean(b))) * periods_per_year
    return beta, alpha


def compare(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    benchmark_equity: pd.Series,
    periods_per_year: float = 252.0,
) -> BenchmarkStats:
    """Relative statistics of ``returns`` versus a benchmark."""
    r, b = returns.align(benchmark_returns, join="inner")
    beta, alpha = beta_alpha(r, b, periods_per_year)
    active = (r - b).to_numpy(dtype=np.float64)
    te = math.nan
    if len(active) > 1:
        te = float(np.std(active, ddof=1) * math.sqrt(periods_per_year))
    ir = math.nan
    if te > 0:
        ir = float(np.mean(active) * periods_per_year / te)
    rv, bv = r.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
    corr = (
        float(np.corrcoef(rv, bv)[0, 1])
        if len(r) > 1 and np.std(rv) > 0 and np.std(bv) > 0
        else math.nan
    )
    return BenchmarkStats(
        beta=beta,
        alpha=alpha,
        correlation=corr,
        tracking_error=te,
        information_ratio=ir,
        total_return=total_return(benchmark_equity),
        cagr=cagr(benchmark_equity, periods_per_year),
        sharpe=sharpe_ratio(b, periods_per_year),
        volatility=annualized_volatility(b, periods_per_year),
        max_drawdown=max_drawdown(benchmark_equity),
    )
