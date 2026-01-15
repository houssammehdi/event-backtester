"""Performance analytics: metrics, round-trip trades, benchmarks and reports."""

from backtester.analytics.benchmark import BenchmarkStats, beta_alpha, buy_and_hold, compare
from backtester.analytics.metrics import (
    Metrics,
    TradeStats,
    annualized_turnover,
    annualized_volatility,
    average_exposure,
    cagr,
    calmar_ratio,
    compute_metrics,
    downside_deviation,
    drawdown_series,
    max_drawdown,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    time_in_market,
    total_return,
    trade_stats,
)
from backtester.analytics.report import format_report
from backtester.analytics.trades import round_trips

__all__ = [
    "BenchmarkStats",
    "Metrics",
    "TradeStats",
    "annualized_turnover",
    "annualized_volatility",
    "average_exposure",
    "beta_alpha",
    "buy_and_hold",
    "cagr",
    "calmar_ratio",
    "compare",
    "compute_metrics",
    "downside_deviation",
    "drawdown_series",
    "format_report",
    "max_drawdown",
    "max_drawdown_duration",
    "round_trips",
    "sharpe_ratio",
    "sortino_ratio",
    "time_in_market",
    "total_return",
    "trade_stats",
]
