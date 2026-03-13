"""The HTML tear sheet and its SVG chart helpers."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import DataFeed, RiskLimits, generate_ohlcv
from backtester.analytics.svg import (
    Series,
    _decimate,
    histogram_chart,
    nice_ticks,
    time_series_chart,
    time_ticks,
)
from backtester.analytics.tearsheet import (
    fmt_money,
    fmt_num,
    fmt_pct,
    monthly_returns,
    rolling_sharpe,
    tear_sheet_html,
)
from backtester.config import BacktestConfig
from backtester.engine import BacktestResult
from backtester.strategies import TimeSeriesMomentum
from tests.conftest import Scripted

VOID = {"meta", "br", "hr", "img", "input", "link", "area", "base", "col", "source", "wbr"}
SVG_EMPTY = {"line", "path", "circle", "rect"}


class _Balance(HTMLParser):
    """Checks that every element is closed in order (void elements aside)."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()


@pytest.fixture(scope="module")
def result() -> BacktestResult:
    feed = DataFeed(generate_ohlcv(n_symbols=3, years=3, seed=4))
    return BacktestConfig().run(feed, TimeSeriesMomentum(lookback=63))


@pytest.fixture(scope="module")
def page(result: BacktestResult) -> str:
    return tear_sheet_html(result, notes=[("Data", "test feed")], n_samples=300)


def test_page_is_self_contained_and_well_formed(page: str) -> None:
    assert page.startswith("<!doctype html>")
    for external in ("<link", "@import", "url(", " src=", "href="):
        assert external not in page
    assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", page)  # only the SVG namespace
    checker = _Balance()
    checker.feed(page)
    assert checker.errors == []
    assert checker.stack == []
    assert page.count("<svg") == page.count("</svg>") >= 7


def test_page_has_every_section_and_both_colour_schemes(page: str) -> None:
    for title in (
        "Equity",
        "Drawdown",
        "Rolling Sharpe ratio (6-month)",
        "Rolling volatility (6-month)",
        "Monthly returns",
        "Daily returns",
        "Trade returns",
        "Exposure",
        "Statistical validation",
        "Bootstrap distribution of the Sharpe ratio",
        "Month-end values of the charts (table view)",
    ):
        assert title in page
    assert "prefers-color-scheme: dark" in page
    assert ':root[data-theme="dark"]' in page
    assert "test feed" in page


def test_headline_numbers_are_the_metrics(result: BacktestResult, page: str) -> None:
    m = result.metrics()
    for text in (
        fmt_pct(m.cagr, 2),
        fmt_num(m.sharpe),
        fmt_pct(-m.max_drawdown),
        fmt_money(m.final_equity),
        fmt_pct(m.total_return, 2),
    ):
        assert f">{text}<" in page


def test_chart_data_blocks_match_their_charts(page: str) -> None:
    blocks = re.findall(r'<script type="application/json" class="chart-data">(.*?)</script>', page)
    assert len(blocks) == 5  # equity, drawdown, two rolling charts, exposure
    for raw in blocks:
        data = json.loads(raw)
        n = len(data["x"])
        assert n == len(data["labels"]) > 10
        assert data["x"] == sorted(data["x"])
        for series in data["series"]:
            assert len(series["y"]) == len(series["text"]) == n


def test_deflated_sharpe_appears_with_trials(result: BacktestResult) -> None:
    trials = [0.01, 0.02, 0.03, 0.04]
    page = tear_sheet_html(result, n_samples=200, trial_sharpes=trials)
    assert "Deflated Sharpe ratio (4 trials)" in page
    assert "corrects for the search" in page
    assert "Deflated Sharpe" not in tear_sheet_html(result, n_samples=200)


def test_degenerate_runs_still_render(tmp_path: Path) -> None:
    """No trades and flat equity, a run shorter than the rolling window, and a halt."""
    feed = DataFeed(generate_ohlcv(n_symbols=2, years=0.3, seed=1))
    idle = BacktestConfig().run(feed, Scripted({}))
    page = tear_sheet_html(idle, n_samples=100)
    assert "Nothing to show." in page  # no trades
    assert "n/a" in page
    halted = BacktestConfig(limits=RiskLimits(max_drawdown=0.02)).run(
        DataFeed(generate_ohlcv(n_symbols=3, years=2, seed=2)), TimeSeriesMomentum(lookback=20)
    )
    assert halted.halted_at is not None
    out = halted.tear_sheet(tmp_path / "halted.html", n_samples=100)
    assert "Kill switch triggered" in out.read_text(encoding="utf-8")


def test_monthly_returns_compound_within_each_month() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2025-03-03"])
    returns = pd.Series([0.10, -0.05, 0.02, 0.01], index=index)
    table = monthly_returns(returns)
    assert list(table.index) == [2024, 2025]
    assert table.loc[2024, "Jan"] == pytest.approx(1.10 * 0.95 - 1)
    assert table.loc[2024, "Feb"] == pytest.approx(0.02)
    assert np.isnan(table.loc[2024, "Mar"])
    assert table.loc[2024, "Year"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)
    assert table.loc[2025, "Year"] == pytest.approx(0.01)


def test_rolling_sharpe_matches_a_direct_computation() -> None:
    rng = np.random.default_rng(0)
    returns = pd.Series(
        rng.normal(0.001, 0.01, 300), index=pd.bdate_range("2024-01-01", periods=300)
    )
    rolled = rolling_sharpe(returns, 50, 252)
    window = returns.iloc[100:150].to_numpy()
    expected = window.mean() / window.std(ddof=1) * np.sqrt(252)
    assert rolled.iloc[149] == pytest.approx(expected)
    assert rolled.iloc[:49].isna().all()


def test_formatting_uses_true_minus_signs_and_no_negative_zero() -> None:
    assert fmt_pct(-0.1234) == "\N{MINUS SIGN}12.3%"
    assert fmt_pct(-0.00001) == "0.0%"
    assert fmt_num(-0.004) == "0.00"
    assert fmt_num(1234.5) == "1,234.50"
    assert fmt_pct(float("nan")) == "n/a"
    assert fmt_money(2_394_940.0) == "2.39M"
    assert fmt_money(15_300.0) == "15.3K"
    assert fmt_money(-1_500_000.0) == "\N{MINUS SIGN}1.50M"


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.0, 1.0, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
        (-0.17, 0.0, [-0.2, -0.15, -0.1, -0.05, 0.0]),
        (1_000_000.0, 2_394_940.0, [1e6, 1.5e6, 2e6, 2.5e6]),
        (3.0, 3.0, [2.6, 2.8, 3.0, 3.2, 3.4]),
    ],
)
def test_nice_ticks_bracket_the_range(lo: float, hi: float, expected: list[float]) -> None:
    ticks = nice_ticks(lo, hi)
    assert ticks == pytest.approx(expected)
    assert ticks[0] <= lo and ticks[-1] >= hi  # noqa: PT018


def test_time_ticks_pick_years_or_months() -> None:
    years = time_ticks(pd.bdate_range("2014-01-02", periods=2520))
    assert [label for _, label in years][:3] == ["2015", "2016", "2017"]
    months = time_ticks(pd.bdate_range("2024-01-15", periods=120))
    assert months[0][1] == "Feb 2024"
    assert all(ts.day == 1 for ts, _ in months)


def test_decimation_keeps_the_extremes_of_every_bucket() -> None:
    rng = np.random.default_rng(1)
    x = np.arange(10_000, dtype=np.float64)
    y = np.cumsum(rng.normal(size=10_000))
    keep = _decimate(x, y, 200)
    assert len(keep) <= 4 * 200
    assert {0, 9_999, int(np.argmin(y)), int(np.argmax(y))} <= set(keep.tolist())
    assert np.all(np.diff(keep) > 0)


def test_chart_helpers_escape_names_and_handle_gaps() -> None:
    index = pd.bdate_range("2024-01-01", periods=30)
    y = np.linspace(1.0, 2.0, 30)
    y[10:12] = np.nan
    html = time_series_chart(
        "A <b> chart", index, [Series("x & y", y, "var(--series-1)")], y_format=fmt_num
    )
    assert "A &lt;b&gt; chart" in html
    assert "x &amp; y" in html
    assert html.count(' d="M') == 1 and "M" in html.split('class="line" d="')[1][1:]  # noqa: PT018
    bars = histogram_chart(
        "h", np.array([-2.0, -1.0, 1.0, 2.0, 3.0]), x_format=fmt_num, diverging=True
    )
    assert "var(--negative)" in bars
    assert "var(--positive)" in bars
    assert bars.count('class="bar"') == 5  # every value in a bin of its own
    assert 'data-tip="2.85 to 3.00: 1 observation"' in bars
