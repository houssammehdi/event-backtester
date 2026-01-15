"""Plain-text performance reports."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtester.engine import BacktestResult


def pct(value: float, digits: int = 2) -> str:
    """Format a fraction as a percentage (``nan`` -> ``n/a``)."""
    if value is None or math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    """Format a number with fixed decimals (``nan`` -> ``n/a``)."""
    if value is None or math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:,.{digits}f}"


def table(rows: Sequence[Sequence[str]], header: Sequence[str] | None = None) -> str:
    """Render rows as a left/right aligned text table."""
    all_rows = [list(header)] if header else []
    all_rows += [list(r) for r in rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(all_rows[0]))]
    lines = []
    for n, r in enumerate(all_rows):
        rest = [c.rjust(w) for c, w in zip(r[1:], widths[1:], strict=True)]
        cells = [r[0].ljust(widths[0]), *rest]
        lines.append("  ".join(cells).rstrip())
        if header and n == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_params(params: dict[str, object]) -> str:
    """``key=value`` list of strategy parameters."""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def format_report(result: BacktestResult, risk_free_rate: float = 0.0) -> str:
    """Build the text report printed by the CLI."""
    m = result.metrics(risk_free_rate=risk_free_rate)
    b = m.benchmark
    t = m.trades
    title = f"Strategy: {result.strategy_name}"
    if result.params:
        title += f" ({format_params(result.params)})"
    lines = [
        title,
        f"Period:   {m.start.date()} -> {m.end.date()}  ({m.n_bars} bars, "
        f"{len(result.prices.columns)} symbols)",
        f"Capital:  {num(m.initial_equity, 0)} -> {num(m.final_equity, 0)}",
        "",
    ]
    perf = [
        ("Total return", pct(m.total_return), pct(b.total_return) if b else "n/a"),
        ("CAGR", pct(m.cagr), pct(b.cagr) if b else "n/a"),
        ("Annual volatility", pct(m.annual_volatility), pct(b.volatility) if b else "n/a"),
        ("Sharpe ratio", num(m.sharpe), num(b.sharpe) if b else "n/a"),
        ("Sortino ratio", num(m.sortino), ""),
        ("Max drawdown", pct(m.max_drawdown), pct(b.max_drawdown) if b else "n/a"),
        ("Max DD duration (bars)", str(m.max_drawdown_duration), ""),
        ("Calmar ratio", num(m.calmar), ""),
    ]
    lines += [table(perf, header=("Performance", "Strategy", "Buy & hold (EW)")), ""]
    if b:
        rel = [
            ("Beta", num(b.beta)),
            ("Alpha (annualised)", pct(b.alpha)),
            ("Correlation", num(b.correlation)),
            ("Tracking error", pct(b.tracking_error)),
            ("Information ratio", num(b.information_ratio)),
        ]
        lines += [table(rel, header=("Versus benchmark", "")), ""]
    trading = [
        ("Round trips (closed)", str(t.n_trades)),
        ("Hit rate", pct(t.hit_rate, 1)),
        ("Profit factor", num(t.profit_factor)),
        ("Average win", num(t.avg_win)),
        ("Average loss", num(t.avg_loss)),
        ("Average bars held", num(t.avg_bars_held, 1)),
        ("Turnover (annual)", pct(m.turnover, 0)),
        ("Average gross exposure", pct(m.exposure, 1)),
        ("Time in market", pct(m.time_in_market, 1)),
    ]
    lines += [table(trading, header=("Trading", "")), ""]
    costs = [
        ("Commission", num(m.total_commission)),
        ("Slippage", num(m.total_slippage)),
        ("Borrow", num(m.total_borrow_cost)),
    ]
    lines += [table(costs, header=("Costs", "")), ""]
    if result.halted_at is not None:
        lines.append(f"Kill-switch triggered at {result.halted_at.date()}: positions flattened.")
    n_risk = sum(1 for a in result.risk_log if a.kind != "kill_switch")
    lines.append(f"Risk actions logged: {n_risk}")
    return "\n".join(lines)
