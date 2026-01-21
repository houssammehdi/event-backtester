"""Equity-curve and drawdown charts (requires the optional ``plot`` extra)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtester.analytics.metrics import drawdown_series

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from backtester.engine import BacktestResult
    from backtester.research.walkforward import WalkForwardResult

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834")
"""Categorical slots 1 and 2: strategy, benchmark."""


def _money(value: float, _: object = None) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}k"
    return f"{value:.0f}"


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "plotting needs matplotlib: pip install 'event-backtester[plot]'"
        ) from exc
    return plt


def plot_equity(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    title: str = "Equity curve",
    subtitle: str = "",
    labels: tuple[str, str] = ("Strategy", "Buy & hold (equal weight)"),
    path: str | Path | None = None,
    log_scale: bool = True,
) -> Figure:
    """Plot an equity curve (optionally against a benchmark) with its drawdown below.

    Two panels share the time axis - equity on top, drawdown underneath - rather than
    a second y-axis. If ``path`` is given the figure is saved as PNG.
    """
    plt = _pyplot()
    fig, (ax_eq, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(10, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
    )
    fig.patch.set_facecolor(SURFACE)
    curves = [(equity, labels[0], SERIES[0])]
    if benchmark is not None:
        curves.append((benchmark, labels[1], SERIES[1]))
    for series, label, color in curves:
        ax_eq.plot(series.index, series.to_numpy(), color=color, linewidth=2, label=label)
        ax_eq.annotate(
            label.split(" (")[0],
            xy=(series.index[-1], float(series.iloc[-1])),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_SECONDARY,
        )
        dd = -drawdown_series(series) * 100
        ax_dd.plot(dd.index, dd.to_numpy(), color=color, linewidth=1.5)
        if color == SERIES[0]:
            ax_dd.fill_between(dd.index, dd.to_numpy(), 0, color=color, alpha=0.15, linewidth=0)
    if log_scale:
        from matplotlib.ticker import LogLocator, NullFormatter

        ax_eq.set_yscale("log")
        ax_eq.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
        ax_eq.yaxis.set_minor_formatter(NullFormatter())
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(_money))
    fig.text(0.1, 0.955, title, fontsize=12, color=TEXT, ha="left", va="bottom")
    if subtitle:
        wrapped = "\n".join(textwrap.wrap(subtitle, width=120))
        fig.text(0.1, 0.95, wrapped, fontsize=8.5, color=TEXT_SECONDARY, ha="left", va="top")
    ax_eq.set_ylabel("Equity", color=TEXT_SECONDARY)
    ax_dd.set_ylabel("Drawdown (%)", color=TEXT_SECONDARY)
    ax_eq.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    ax_eq.margins(x=0.01)
    fig.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.07)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=110, facecolor=SURFACE)
    figure: Figure = fig
    return figure


def plot_result(result: BacktestResult, path: str | Path | None = None) -> Figure:
    """Equity and drawdown of a backtest versus the equal-weight buy-and-hold benchmark."""
    params = ", ".join(f"{k}={v}" for k, v in result.params.items())
    return plot_equity(
        result.equity,
        result.benchmark_equity(),
        title=f"Strategy: {result.strategy_name}",
        subtitle=params,
        path=path,
    )


def plot_walk_forward(
    result: WalkForwardResult,
    benchmark: pd.Series | None = None,
    path: str | Path | None = None,
) -> Figure:
    """Stitched out-of-sample equity (optionally with a benchmark over the same bars)."""
    return plot_equity(
        result.oos_equity,
        benchmark,
        title="Walk-forward: stitched out-of-sample equity",
        subtitle="Each segment traded with parameters fitted only on the preceding window.",
        labels=("Out-of-sample", "Buy & hold (equal weight)"),
        path=path,
    )
