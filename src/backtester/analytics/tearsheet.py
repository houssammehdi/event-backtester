"""A self-contained HTML tear sheet of one backtest.

One file with inline CSS, inline SVG charts and a small inline script (tooltips), no
external resources: equity and drawdown, rolling Sharpe ratio and volatility, a
monthly-returns heatmap, the distribution of daily returns and of trade returns,
exposure, the metrics tables, and the statistical validation of the result
(stationary-bootstrap intervals, PSR, minimum track record length, and the deflated
Sharpe ratio when the trials of a search are supplied). The page follows the
reader's light or dark colour scheme.
"""

from __future__ import annotations

import datetime as _dt
import html
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.analytics.metrics import drawdown_series
from backtester.analytics.report import format_params
from backtester.analytics.svg import Overlay, Series, histogram_chart, time_series_chart
from backtester.validation.report import StrategyValidation, validate_returns

if TYPE_CHECKING:
    from backtester.engine import BacktestResult

FloatArray = npt.NDArray[np.float64]
HALF = 520.0
"""SVG width of the charts shown two per row."""
MINUS = "\N{MINUS SIGN}"
SEPARATOR = " \N{MIDDLE DOT} "
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
HEAT_MAX = 0.7
"""Strongest heatmap tint (share of the pole colour); keeps cell text >= 4.5:1."""


# ---------------------------------------------------------------------- formatting
def _signed(text: str) -> str:
    return text.replace("-", MINUS)


def _unsigned_zero(value: float, digits: int) -> float:
    """``value``, or 0.0 when it rounds to zero (so that no "-0.0" is printed)."""
    return 0.0 if round(value, digits) == 0 else value


def fmt_pct(value: float, digits: int = 1) -> str:
    """``0.1234 -> "12.3%"`` with a true minus sign; NaN -> ``"n/a"``."""
    if not math.isfinite(value):
        return "n/a" if math.isnan(value) else ("inf" if value > 0 else f"{MINUS}inf")
    return _signed(f"{_unsigned_zero(value * 100, digits):.{digits}f}%")


def fmt_num(value: float, digits: int = 2) -> str:
    """Fixed decimals with thousands separators and a true minus sign."""
    if not math.isfinite(value):
        return "n/a" if math.isnan(value) else ("inf" if value > 0 else f"{MINUS}inf")
    return _signed(f"{_unsigned_zero(value, digits):,.{digits}f}")


def fmt_money(value: float) -> str:
    """Compact amount: ``2394940 -> "2.39M"``, ``15300 -> "15.3K"``."""
    if not math.isfinite(value):
        return "n/a"
    size = abs(value)
    for bound, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if size >= bound:
            return _signed(f"{value / bound:.{2 if size < 10 * bound else 1}f}{suffix}")
    return _signed(f"{value:.0f}")


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------- statistics
def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    """Annualised Sharpe ratio over a trailing window (NaN until it is full)."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=1)
    out: pd.Series = (mean / std.where(std > 0) * math.sqrt(periods_per_year)).rename("sharpe")
    return out


def rolling_volatility(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    """Annualised standard deviation over a trailing window."""
    out: pd.Series = returns.rolling(window).std(ddof=1) * math.sqrt(periods_per_year)
    return out.rename("volatility")


def monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """Compounded returns per calendar month: one row per year, 12 months plus ``Year``.

    Months without any return are NaN; ``Year`` compounds the months that exist.
    """
    idx = pd.DatetimeIndex(returns.index)
    frame = pd.DataFrame(
        {"year": idx.year, "month": idx.month, "growth": 1.0 + returns.to_numpy(dtype=np.float64)}
    )
    growth = frame.pivot_table(index="year", columns="month", values="growth", aggfunc="prod")
    table = growth.reindex(columns=range(1, 13)) - 1.0
    table.columns = list(MONTHS)
    table.index.name = "year"
    yearly = (1.0 + returns).groupby(idx.year).prod() - 1.0
    table["Year"] = yearly
    return table


def _normal_pdf(mean: float, std: float) -> Overlay:
    def pdf(x: FloatArray) -> FloatArray:
        out: FloatArray = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
        return out

    return Overlay("Normal, same mean and volatility", pdf, "var(--context)")


# ---------------------------------------------------------------------- page pieces
def _tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<p class="note">{_esc(note)}</p>' if note else ""
    return (
        f'<div class="tile"><p class="label">{_esc(label)}</p>'
        f'<p class="value">{_esc(value)}</p>{note_html}</div>'
    )


def _table(
    rows: Iterable[Sequence[str]], header: Sequence[str], *, caption: str = "", cls: str = ""
) -> str:
    head = "".join(
        f'<th scope="col"{" class=num" if k else ""}>{_esc(h)}</th>' for k, h in enumerate(header)
    )
    body = []
    for row in rows:
        cells = f'<th scope="row">{_esc(row[0])}</th>'
        cells += "".join(f'<td class="num">{_esc(c)}</td>' for c in row[1:])
        body.append(f"<tr>{cells}</tr>")
    cap = f"<caption>{_esc(caption)}</caption>" if caption else ""
    return (
        f'<table class="{cls or "metrics"}">{cap}<thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _heatmap(table: pd.DataFrame) -> str:
    """Monthly returns as an HTML table tinted on the diverging scale."""
    months = table[list(MONTHS)].to_numpy(dtype=np.float64)
    years = table["Year"].to_numpy(dtype=np.float64)
    scale_m = float(np.nanmax(np.abs(months))) if np.isfinite(months).any() else 1.0
    scale_y = float(np.nanmax(np.abs(years))) if np.isfinite(years).any() else 1.0

    def cell(value: float, scale: float, cls: str = "") -> str:
        if not math.isfinite(value):
            return f'<td class="empty {cls}"></td>'
        tint = HEAT_MAX * min(abs(value) / scale, 1.0) if scale > 0 else 0.0
        percent = _unsigned_zero(value * 100, 1)
        sign = "pos" if percent > 0 else "neg" if percent < 0 else "zero"
        text = _signed(f"{percent:+.1f}") if percent else "0.0"
        return f'<td class="{sign} {cls}" style="--t:{tint:.3f}">{text}</td>'

    head = "".join(f'<th scope="col">{m}</th>' for m in (*MONTHS, "Year"))
    rows = []
    for year, m_row, y_value in zip(table.index, months, years, strict=True):
        cells = "".join(cell(float(v), scale_m) for v in m_row)
        cells += cell(float(y_value), scale_y, "year")
        rows.append(f'<tr><th scope="row">{year}</th>{cells}</tr>')
    return (
        '<figure class="chart"><figcaption><h3>Monthly returns</h3>'
        '<p class="sub">Compounded return per calendar month, in percent; the last column '
        "compounds the year. Blue is a gain, red a loss; months and years have their own "
        "colour scales.</p></figcaption>"
        '<div class="plot"><table class="heat"><thead><tr><th scope="col">'
        f"</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div></figure>"
    )


def _data_table(frame: pd.DataFrame) -> str:
    formats = {
        "Equity": fmt_money,
        "Buy & hold": fmt_money,
        "Drawdown": fmt_pct,
        "Rolling Sharpe": fmt_num,
        "Rolling volatility": fmt_pct,
        "Gross exposure": fmt_pct,
        "Net exposure": fmt_pct,
    }
    months = pd.DatetimeIndex(frame.index).strftime("%Y-%m")
    columns = [str(c) for c in frame.columns]
    values = frame.to_numpy(dtype=np.float64)
    rows = [
        [month, *(formats[c](float(v)) for c, v in zip(columns, row, strict=True))]
        for month, row in zip(months, values, strict=True)
    ]
    table = _table(rows, ["Month", *columns], cls="metrics data")
    return (
        '<details class="card"><summary>Month-end values of the charts (table view)'
        f'</summary><div class="plot">{table}</div></details>'
    )


def _validation(validation: StrategyValidation, deflated: bool) -> tuple[str, str]:
    """(tables and notes, bootstrap histogram) of the validation section."""
    boot, sharpe = validation.bootstrap, validation.sharpe
    level = f"{boot.confidence:.0%}"
    rows = []
    for key, label, fmt, sign in (
        ("sharpe", "Sharpe ratio (annualised)", fmt_num, 1.0),
        ("cagr", "CAGR", fmt_pct, 1.0),
        ("max_drawdown", "Max drawdown", fmt_pct, -1.0),  # shown as a loss
        ("volatility", "Annual volatility", fmt_pct, 1.0),
    ):
        iv = boot.intervals[key]
        low, high = sorted((sign * iv.lower, sign * iv.upper))
        rows.append(
            (label, fmt(sign * iv.estimate), f"{fmt(low)} to {fmt(high)}", fmt(iv.std_error))
        )
    table = _table(rows, ["Statistic", "Estimate", f"{level} interval", "Bootstrap s.e."])
    years = sharpe.min_track_record_years
    mintrl = "never" if math.isinf(years) else "n/a" if math.isnan(years) else f"{years:.1f} years"
    rows2 = [
        ("Probabilistic Sharpe ratio, P(true Sharpe > 0)", fmt_num(sharpe.psr, 3)),
        (f"Minimum track record ({level})", mintrl),
        ("Track record in the sample", f"{sharpe.track_record_years:.1f} years"),
        (
            "Skewness / excess kurtosis of returns",
            f"{fmt_num(sharpe.skew)} / {fmt_num(sharpe.kurtosis - 3)}",
        ),
    ]
    if sharpe.dsr is not None:
        rows2.insert(
            1, (f"Deflated Sharpe ratio ({sharpe.n_trials} trials)", fmt_num(sharpe.dsr, 3))
        )
    table2 = _table(rows2, ["Sharpe-ratio inference", ""])
    sr = boot.intervals["sharpe"]
    verdict = (
        f"The {level} interval of the Sharpe ratio excludes zero."
        if sr.lower > 0
        else f"The {level} interval of the Sharpe ratio includes zero: this sample does not "
        "rule out a strategy without skill."
    )
    if math.isfinite(years):
        enough = sharpe.track_record_years >= years
        verdict += (
            " The sample is longer than the minimum track record."
            if enough
            else " The sample is shorter than the minimum track record needed to call the "
            f"Sharpe ratio positive at {level}."
        )
    deflation = (
        "The deflated Sharpe ratio corrects for the search the parameters came from."
        if deflated
        else "No correction for parameter selection is applied: if these parameters came "
        "out of a search, run backtest validate (deflated Sharpe, PBO, SPA)."
    )
    notes = (
        f'<p class="verdict">{_esc(verdict)}</p>'
        f'<p class="fine">Stationary bootstrap (Politis &amp; Romano 1994) with '
        f"{boot.n_samples:,} resamples and a mean block length of {boot.block_length:.1f} "
        f"bars (Politis &amp; White 2004, corrected by Patton, Politis &amp; White 2009); "
        f"{boot.method} intervals. PSR and minimum track record length after Bailey &amp; "
        f"López de Prado (2012), using the skewness and kurtosis of the returns. "
        f"{_esc(deflation)}</p>"
    )
    marks = [(sr.lower, f"{level} low"), (sr.upper, f"{level} high")]
    if boot.samples["sharpe"].min() < 0 < boot.samples["sharpe"].max():
        marks.append((0.0, "0"))
    chart = histogram_chart(
        "Bootstrap distribution of the Sharpe ratio",
        boot.samples["sharpe"],
        x_format=lambda v: fmt_num(v, 1),
        bins=40,
        subtitle=f"{boot.n_samples:,} stationary-bootstrap resamples of the daily returns; "
        f"the lines mark the {level} interval.",
        name="Resamples",
        marks=marks,
        unit="resamples",
        height=220,
        width=HALF,
    )
    return table + table2 + notes, chart


# ---------------------------------------------------------------------- the page
def tear_sheet_html(
    result: BacktestResult,
    *,
    title: str | None = None,
    notes: Sequence[tuple[str, str]] = (),
    rolling_window: int = 126,
    n_samples: int = 2000,
    confidence: float = 0.95,
    trial_sharpes: npt.ArrayLike | None = None,
    risk_free_rate: float = 0.0,
    seed: int = 0,
) -> str:
    """Render the tear sheet of ``result`` as one self-contained HTML document.

    Args:
        result: The backtest.
        title: Page title (default: the strategy name).
        notes: Extra ``(label, text)`` lines for the header, e.g. the data and costs.
        rolling_window: Bars of the rolling Sharpe and volatility windows.
        n_samples: Bootstrap resamples for the validation section.
        confidence: Interval coverage and minimum-track-record confidence.
        trial_sharpes: Per-period Sharpe ratios of every configuration of the search
            that produced this one; adds the deflated Sharpe ratio.
        risk_free_rate: Annual rate for the Sharpe and Sortino ratios of the tables.
        seed: Bootstrap seed.
    """
    from backtester import __version__

    m = result.metrics(risk_free_rate=risk_free_rate)
    b = m.benchmark
    ppy = result.periods_per_year
    returns = result.returns
    equity = result.equity
    index = pd.DatetimeIndex(equity.index)
    bench_equity = result.benchmark_equity()
    bench_returns = bench_equity.pct_change().iloc[1:]
    validation = validate_returns(
        returns,
        periods_per_year=ppy,
        n_samples=n_samples,
        confidence=confidence,
        trial_sharpes=trial_sharpes,
        seed=seed,
    )
    name = title or result.strategy_name
    facts = [
        (
            "Period",
            SEPARATOR.join(
                (
                    f"{m.start.date()} to {m.end.date()}",
                    f"{m.n_bars:,} bars",
                    f"{len(result.prices.columns)} symbols",
                )
            ),
        )
    ]
    if result.params:
        facts.insert(0, ("Parameters", format_params(result.params)))
    facts += [(label, text) for label, text in notes]
    sr = validation.bootstrap.intervals["sharpe"]
    level = f"{confidence:.0%}"
    tiles = [
        _tile("CAGR", fmt_pct(m.cagr, 2), f"Buy & hold {fmt_pct(b.cagr, 2)}" if b else ""),
        _tile(
            "Sharpe ratio",
            fmt_num(m.sharpe),
            f"{level} CI {fmt_num(sr.lower)} to {fmt_num(sr.upper)}",
        ),
        _tile(
            "Max drawdown", fmt_pct(-m.max_drawdown), f"Longest {m.max_drawdown_duration:,} bars"
        ),
        _tile("Volatility", fmt_pct(m.annual_volatility), "Annualised"),
        _tile("PSR", fmt_num(validation.sharpe.psr, 3), "P(true Sharpe > 0)"),
        _tile("Final equity", fmt_money(m.final_equity), f"From {fmt_money(m.initial_equity)}"),
    ]
    drawdown = -drawdown_series(equity).to_numpy(dtype=np.float64)  # losses are negative
    bench_dd = -drawdown_series(bench_equity).to_numpy(dtype=np.float64)
    roll_sr = rolling_sharpe(returns, rolling_window, ppy).reindex(index)
    roll_vol = rolling_volatility(returns, rolling_window, ppy).reindex(index)
    bench_vol = rolling_volatility(bench_returns, rolling_window, ppy).reindex(index)
    exposure = result.exposure.div(equity.where(equity > 0), axis=0)
    months = rolling_window / (ppy / 12)
    window_text = (
        f"{months:.0f}-month" if abs(months - round(months)) < 0.05 else f"{rolling_window}-bar"
    )
    charts_top = [
        time_series_chart(
            "Equity",
            index,
            [
                Series(result.strategy_name, equity.to_numpy(dtype=np.float64), "var(--series-1)"),
                Series("Buy & hold", bench_equity.to_numpy(dtype=np.float64), "var(--context)"),
            ],
            y_format=fmt_money,
            height=300,
            subtitle="Marked to the close after costs, against an equal-weight buy-and-hold "
            "of the same universe with the same starting capital.",
        ),
        time_series_chart(
            "Drawdown",
            index,
            [
                Series(result.strategy_name, drawdown, "var(--negative)", area=True),
                Series("Buy & hold", bench_dd, "var(--context)"),
            ],
            y_format=lambda v: fmt_pct(v, 0),
            height=200,
            baseline=0.0,
            subtitle="Fall from the running equity peak.",
        ),
    ]
    rolling = [
        time_series_chart(
            f"Rolling Sharpe ratio ({window_text})",
            index,
            [Series(result.strategy_name, roll_sr.to_numpy(dtype=np.float64), "var(--series-1)")],
            y_format=fmt_num,
            tick_format=lambda v: fmt_num(v, 0 if v == round(v) else 1),
            height=240,
            baseline=0.0,
            subtitle=f"Annualised, over the trailing {rolling_window} bars.",
            width=HALF,
        ),
        time_series_chart(
            f"Rolling volatility ({window_text})",
            index,
            [
                Series(
                    result.strategy_name, roll_vol.to_numpy(dtype=np.float64), "var(--series-1)"
                ),
                Series("Buy & hold", bench_vol.to_numpy(dtype=np.float64), "var(--context)"),
            ],
            y_format=lambda v: fmt_pct(v, 0),
            height=240,
            include_zero=True,
            subtitle=f"Annualised standard deviation over the trailing {rolling_window} bars.",
            width=HALF,
        ),
    ]
    r = returns.to_numpy(dtype=np.float64)
    std = float(np.std(r, ddof=1)) if len(r) > 1 else math.nan
    var95 = float(np.quantile(r, 0.05)) if len(r) else math.nan
    cvar95 = float(r[r <= var95].mean()) if len(r) else math.nan
    dist_note = (
        f"Skewness {fmt_num(validation.sharpe.skew)}, excess kurtosis "
        f"{fmt_num(validation.sharpe.kurtosis - 3)}; on the worst 5% of days the loss was "
        f"at least {fmt_pct(-var95, 2)} and {fmt_pct(-cvar95, 2)} on average."
    )
    trades = result.trades()
    closed = trades.loc[trades["status"] == "closed"] if len(trades) else trades
    trade_returns = closed["return"].to_numpy(dtype=np.float64) if len(closed) else np.zeros(0)
    t = m.trades
    distributions = [
        histogram_chart(
            "Daily returns",
            r,
            x_format=lambda v: fmt_pct(v, 1),
            bins=50,
            subtitle=dist_note,
            name="Days",
            overlay=_normal_pdf(float(np.mean(r)), std) if math.isfinite(std) and std > 0 else None,
            unit="days",
            width=HALF,
        ),
        histogram_chart(
            "Trade returns",
            trade_returns,
            x_format=lambda v: fmt_pct(v, 0),
            bins=30,
            subtitle=f"{t.n_trades:,} closed round trips, net of commissions, relative to the "
            f"entry notional; hit rate {fmt_pct(t.hit_rate)}, profit factor "
            f"{fmt_num(t.profit_factor)}.",
            diverging=True,
            unit="trades",
            width=HALF,
        ),
    ]
    exposure_chart = time_series_chart(
        "Exposure",
        index,
        [
            Series("Gross", exposure["gross"].to_numpy(dtype=np.float64), "var(--series-1)"),
            Series("Net", exposure["net"].to_numpy(dtype=np.float64), "var(--series-2)"),
        ],
        y_format=lambda v: fmt_pct(v, 0),
        height=220,
        baseline=0.0,
        subtitle="Sum of absolute (gross) and signed (net) position values, as a share of equity.",
    )
    perf = [
        ("Total return", fmt_pct(m.total_return, 2), fmt_pct(b.total_return, 2) if b else "n/a"),
        ("CAGR", fmt_pct(m.cagr, 2), fmt_pct(b.cagr, 2) if b else "n/a"),
        (
            "Annual volatility",
            fmt_pct(m.annual_volatility, 2),
            fmt_pct(b.volatility, 2) if b else "n/a",
        ),
        ("Sharpe ratio", fmt_num(m.sharpe), fmt_num(b.sharpe) if b else "n/a"),
        ("Sortino ratio", fmt_num(m.sortino), ""),
        ("Max drawdown", fmt_pct(-m.max_drawdown, 2), fmt_pct(-b.max_drawdown, 2) if b else "n/a"),
        ("Longest drawdown (bars)", f"{m.max_drawdown_duration:,}", ""),
        ("Calmar ratio", fmt_num(m.calmar), ""),
    ]
    relative = []
    if b:
        relative = [
            ("Beta", fmt_num(b.beta)),
            ("Alpha (annualised)", fmt_pct(b.alpha, 2)),
            ("Correlation", fmt_num(b.correlation)),
            ("Tracking error", fmt_pct(b.tracking_error, 2)),
            ("Information ratio", fmt_num(b.information_ratio)),
        ]
    trading = [
        ("Closed round trips", f"{t.n_trades:,}"),
        ("Hit rate", fmt_pct(t.hit_rate)),
        ("Profit factor", fmt_num(t.profit_factor)),
        ("Average win", fmt_num(t.avg_win)),
        ("Average loss", fmt_num(t.avg_loss)),
        ("Average bars held", fmt_num(t.avg_bars_held, 1)),
        ("Turnover (annual)", fmt_pct(m.turnover, 0)),
        ("Average gross exposure", fmt_pct(m.exposure)),
        ("Time in market", fmt_pct(m.time_in_market)),
        ("Commission", fmt_num(m.total_commission)),
        ("Slippage", fmt_num(m.total_slippage)),
        ("Borrow fees", fmt_num(m.total_borrow_cost)),
    ]
    left = _table(perf, ["Performance", "Strategy", "Buy & hold"])
    if relative:
        left += _table(relative, ["Against buy & hold", ""])
    tables = [f'<div class="card">{left}</div>']
    tables.append(f'<div class="card">{_table(trading, ["Trading and costs", ""])}</div>')
    valid_tables, boot_chart = _validation(validation, trial_sharpes is not None)
    month_end = pd.DataFrame(
        {
            "Equity": equity,
            "Buy & hold": bench_equity,
            "Drawdown": pd.Series(drawdown, index=index),
            "Rolling Sharpe": roll_sr,
            "Rolling volatility": roll_vol,
            "Gross exposure": exposure["gross"],
            "Net exposure": exposure["net"],
        }
    )
    month_end = month_end.groupby([index.year, index.month]).tail(1)
    halted = (
        f'<p class="alert">Kill switch triggered on {result.halted_at.date()}: positions '
        "were flattened and the strategy stopped.</p>"
        if result.halted_at is not None
        else ""
    )
    fact_rows = "".join(f"<div><dt>{_esc(k)}</dt><dd>{_esc(v)}</dd></div>" for k, v in facts)
    generated = _dt.datetime.now(tz=_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    body = f"""
<header>
  <p class="eyebrow">Backtest tear sheet</p>
  <h1>{_esc(name)}</h1>
  <dl class="facts">{fact_rows}</dl>{halted}
</header>
<section class="kpis" aria-label="Headline figures">{"".join(tiles)}</section>
<section class="stack">{"".join(charts_top)}</section>
<section class="pair">{"".join(rolling)}</section>
<section class="stack">{_heatmap(monthly_returns(returns))}</section>
<section class="pair">{"".join(distributions)}</section>
<section class="stack">{exposure_chart}</section>
<section class="tables"><h2>Metrics</h2><div class="pair">{"".join(tables)}</div></section>
<section class="tables"><h2>Statistical validation</h2>
  <div class="pair"><div class="card">{valid_tables}</div>{boot_chart}</div></section>
<section class="stack">{_data_table(month_end)}</section>
<footer>
  <p>event-backtester {_esc(__version__)} · generated {generated}. Orders decided at a
  bar's close execute on later bars only (market orders at the next open); the
  intervals quantify sampling uncertainty, not the risk of a flawed model or of data
  mining.</p>
</footer>"""
    return _PAGE.format(title=_esc(name), css=_CSS, body=body, script=_SCRIPT)


def write_tear_sheet(result: BacktestResult, path: str | Path, **kwargs: Any) -> Path:
    """Write :func:`tear_sheet_html` to ``path`` (UTF-8) and return the path."""
    out = Path(path)
    out.write_text(tear_sheet_html(result, **kwargs), encoding="utf-8")
    return out


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>{body}
</main>
<script>{script}</script>
</body>
</html>
"""

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --context: #898781;
  --positive: #2a78d6; --negative: #e34948; --div-mid: #f0efec; --alert: #d03b3b;
  --shadow: 0 6px 24px rgba(11, 11, 11, 0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5; --series-2: #d95926; --context: #898781;
    --positive: #3987e5; --negative: #e66767; --div-mid: #383835; --alert: #d03b3b;
    --shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5; --series-2: #d95926; --context: #898781;
  --positive: #3987e5; --negative: #e66767; --div-mid: #383835; --alert: #d03b3b;
  --shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 1160px; margin: 0 auto; padding: 36px 16px 48px; }
header { margin: 0 4px 24px; }
.eyebrow { margin: 0; color: var(--ink-2); font-size: 13px; letter-spacing: 0.02em; }
h1 { margin: 2px 0 6px; font-size: 28px; font-weight: 600; letter-spacing: -0.01em; }
h2 { margin: 8px 4px 12px; font-size: 18px; font-weight: 600; }
h3 { margin: 0; font-size: 15px; font-weight: 600; }
.facts { margin: 0; display: grid; gap: 2px; }
.facts div { display: flex; gap: 8px; }
.facts dt { color: var(--ink-2); min-width: 88px; }
.facts dd { margin: 0; }
.alert { margin: 10px 0 0; color: var(--alert); font-weight: 600; }
section { margin: 0 0 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.tile, .chart, .card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
}
.tile { padding: 14px 16px; }
.tile p { margin: 0; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; line-height: 1.25; margin-top: 2px; }
.tile .note { color: var(--ink-2); font-size: 12px; margin-top: 2px; }
.stack { display: grid; gap: 16px; }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; }
.chart { margin: 0; padding: 16px 18px 12px; min-width: 0; }
.card { padding: 16px 18px; min-width: 0; overflow-x: auto; }
.sub { margin: 2px 0 0; color: var(--ink-2); font-size: 13px; max-width: 72ch; }
.legend {
  list-style: none; display: flex; flex-wrap: wrap; gap: 4px 18px;
  margin: 10px 0 0; padding: 0; font-size: 12px; color: var(--ink-2);
}
.legend li { display: flex; align-items: center; gap: 6px; }
.key { display: inline-block; background: var(--c); }
.key.line { width: 16px; height: 2px; border-radius: 1px; }
.key.rect { width: 10px; height: 10px; border-radius: 2px; }
.plot { overflow-x: auto; margin-top: 8px; }
.plot svg { display: block; width: 100%; height: auto; overflow: visible; }
.plot svg:focus-visible {
  outline: 2px solid var(--series-1); outline-offset: 4px; border-radius: 4px;
}
svg text { font-family: inherit; }
svg .grid { stroke: var(--grid); stroke-width: 1; vector-effect: non-scaling-stroke; }
svg .axis { stroke: var(--axis); stroke-width: 1; vector-effect: non-scaling-stroke; }
svg .tick { fill: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
svg .line {
  fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round;
  vector-effect: non-scaling-stroke;
}
svg .area { opacity: 0.12; stroke: none; }
svg .end { fill: var(--ink); font-size: 12px; font-weight: 600; }
svg .end-name { fill: var(--ink-2); font-size: 11px; }
svg .bar { transition: opacity 0.1s; }
svg .bar:hover, svg .bar:focus { opacity: 0.72; outline: none; }
svg .mark { stroke: var(--ink-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
svg .mark-label { fill: var(--ink-2); font-size: 11px; }
svg .cross { stroke: var(--ink-2); stroke-width: 1; vector-effect: non-scaling-stroke; }
svg .dot { stroke: var(--surface); stroke-width: 2; vector-effect: non-scaling-stroke; }
.empty { color: var(--ink-2); padding: 24px 0; margin: 0; }
.tip {
  position: fixed; z-index: 10; pointer-events: none; min-width: 150px;
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  border-radius: 8px; box-shadow: var(--shadow); padding: 8px 10px; font-size: 12px;
}
.tip .date { color: var(--ink-2); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 8px; }
.tip .row .key { width: 12px; height: 2px; }
.tip .row .name { color: var(--ink-2); }
.tip .row .v {
  margin-left: auto; padding-left: 12px; font-weight: 600; font-variant-numeric: tabular-nums;
}
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
caption { text-align: left; font-weight: 600; padding-bottom: 6px; }
table.metrics { margin-bottom: 4px; }
table.metrics th, table.metrics td { padding: 6px 4px; text-align: left; font-weight: 400; }
table.metrics thead th {
  color: var(--ink-2); font-weight: 600; border-bottom: 1px solid var(--axis);
}
table.metrics tbody tr + tr > * { border-top: 1px solid var(--grid); }
table.metrics .num { text-align: right; white-space: nowrap; }
.card table + table { margin-top: 16px; }
.verdict { margin: 14px 0 6px; font-weight: 600; }
.fine { margin: 0; color: var(--ink-2); font-size: 12px; }
table.heat { border-collapse: separate; border-spacing: 2px; min-width: 720px; font-size: 12px; }
table.heat th { color: var(--ink-2); font-weight: 400; padding: 4px 6px; text-align: right; }
table.heat thead th { font-weight: 600; }
table.heat td {
  text-align: right; padding: 6px 7px; border-radius: 4px; color: var(--ink);
  background: var(--div-mid);
}
table.heat td.pos {
  background: color-mix(in oklab, var(--positive) calc(var(--t) * 100%), var(--div-mid));
}
table.heat td.neg {
  background: color-mix(in oklab, var(--negative) calc(var(--t) * 100%), var(--div-mid));
}
table.heat td.empty { background: transparent; }
table.heat td.year { font-weight: 600; }
table.heat tr:hover th { color: var(--ink); }
details summary { cursor: pointer; font-weight: 600; }
details .plot { max-height: 420px; overflow: auto; }
table.data { min-width: 720px; font-size: 12px; }
footer { margin: 28px 4px 0; color: var(--ink-2); font-size: 12px; max-width: 90ch; }
@media (max-width: 520px) {
  .pair { grid-template-columns: 1fr; }
  h1 { font-size: 24px; }
  table.metrics .num { white-space: normal; }
}
@media print {
  body { background: #ffffff; }
  .chart, .card, .tile { break-inside: avoid; box-shadow: none; }
  details .plot { max-height: none; }
}
"""

_SCRIPT = """
(() => {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  document.body.appendChild(tip);
  const place = (x, y) => {
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (top + h > innerHeight - 8) top = y - h - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  };
  const hide = () => { tip.hidden = true; };
  const line = (cls, text) => {
    const el = document.createElement("div");
    el.className = cls;
    el.textContent = text;
    return el;
  };
  document.querySelectorAll("figure.chart").forEach((fig) => {
    const svg = fig.querySelector("svg");
    const holder = fig.querySelector("script.chart-data");
    if (svg && holder) {
      const d = JSON.parse(holder.textContent);
      const layer = svg.querySelector("g.hover");
      const cross = layer.querySelector(".cross");
      cross.setAttribute("y1", d.top);
      cross.setAttribute("y2", d.bottom);
      const dots = d.series.map((s) => {
        const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        c.setAttribute("r", 4);
        c.setAttribute("class", "dot");
        c.style.fill = s.color;
        layer.appendChild(c);
        return c;
      });
      let current = d.x.length - 1;
      const nearest = (clientX) => {
        const p = svg.createSVGPoint();
        p.x = clientX;
        p.y = 0;
        const x = p.matrixTransform(svg.getScreenCTM().inverse()).x;
        let lo = 0, hi = d.x.length - 1;
        while (hi - lo > 1) {
          const mid = (lo + hi) >> 1;
          if (d.x[mid] < x) lo = mid; else hi = mid;
        }
        return x - d.x[lo] < d.x[hi] - x ? lo : hi;
      };
      const show = (i, clientX, clientY) => {
        current = i;
        const x = d.x[i];
        cross.setAttribute("x1", x);
        cross.setAttribute("x2", x);
        d.series.forEach((s, k) => {
          const y = s.y[i];
          if (y === null) { dots[k].setAttribute("visibility", "hidden"); return; }
          dots[k].removeAttribute("visibility");
          dots[k].setAttribute("cx", x);
          dots[k].setAttribute("cy", y);
        });
        layer.setAttribute("visibility", "visible");
        tip.replaceChildren(line("date", d.labels[i]));
        d.series.forEach((s) => {
          const row = document.createElement("div");
          row.className = "row";
          const key = document.createElement("span");
          key.className = "key";
          key.style.background = s.color;
          row.append(key, line("name", s.name), line("v", s.text[i]));
          tip.appendChild(row);
        });
        tip.hidden = false;
        place(clientX, clientY);
      };
      const atIndex = (i) => {
        const p = svg.createSVGPoint();
        p.x = d.x[i];
        p.y = d.top;
        const q = p.matrixTransform(svg.getScreenCTM());
        show(i, q.x, q.y);
      };
      const leave = () => { layer.setAttribute("visibility", "hidden"); hide(); };
      svg.addEventListener("pointermove", (e) => show(nearest(e.clientX), e.clientX, e.clientY));
      svg.addEventListener("pointerleave", leave);
      svg.addEventListener("focus", () => atIndex(current));
      svg.addEventListener("blur", leave);
      svg.addEventListener("keydown", (e) => {
        const step = e.shiftKey ? 10 : 1;
        if (e.key === "ArrowLeft") current = Math.max(0, current - step);
        else if (e.key === "ArrowRight") current = Math.min(d.x.length - 1, current + step);
        else return;
        e.preventDefault();
        atIndex(current);
      });
    }
    fig.querySelectorAll("[data-tip]").forEach((el) => {
      const text = el.getAttribute("data-tip");
      const on = (x, y) => {
        tip.replaceChildren(line("v", text));
        tip.hidden = false;
        place(x, y);
      };
      el.addEventListener("pointermove", (e) => on(e.clientX, e.clientY));
      el.addEventListener("pointerleave", hide);
      el.addEventListener("focus", () => {
        const r = el.getBoundingClientRect();
        on(r.right, r.top);
      });
      el.addEventListener("blur", hide);
    });
  });
})();
"""
