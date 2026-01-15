"""Metrics versus values computed by hand."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester.analytics import (
    annualized_turnover,
    annualized_volatility,
    average_exposure,
    beta_alpha,
    buy_and_hold,
    cagr,
    calmar_ratio,
    compare,
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

EQUITY = [100.0, 110.0, 99.0, 120.0]
RETURNS = [0.1, -0.1, 120.0 / 99.0 - 1.0]


def test_return_metrics() -> None:
    assert total_return(EQUITY) == pytest.approx(0.2)
    # 3 return periods with 3 periods per year = 1 year
    assert cagr(EQUITY, periods_per_year=3) == pytest.approx(0.2)
    assert cagr(EQUITY, periods_per_year=1.5) == pytest.approx(1.2**0.5 - 1)
    assert cagr([100.0, 0.0], 1) == -1.0


def test_volatility_sharpe_sortino() -> None:
    r = np.array(RETURNS)
    mean = r.mean()
    sd = math.sqrt(((r - mean) ** 2).sum() / 2)
    assert annualized_volatility(r, 252) == pytest.approx(sd * math.sqrt(252))
    assert sharpe_ratio(r, 252) == pytest.approx(mean / sd * math.sqrt(252))
    rf = 0.03
    excess = r - rf / 252
    assert sharpe_ratio(r, 252, rf) == pytest.approx(
        excess.mean() / excess.std(ddof=1) * math.sqrt(252)
    )
    dd = math.sqrt((0.1**2) / 3)  # only the -10% period falls short of zero
    assert downside_deviation(r) == pytest.approx(dd)
    assert sortino_ratio(r, 252) == pytest.approx(mean / dd * math.sqrt(252))


def test_undefined_ratios() -> None:
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))
    assert sortino_ratio([0.01, 0.02]) == math.inf
    assert math.isnan(sharpe_ratio([0.01]))
    assert math.isnan(calmar_ratio(0.1, 0.0))


def test_drawdowns() -> None:
    assert max_drawdown(EQUITY) == pytest.approx(0.1)  # 110 -> 99
    assert max_drawdown_duration(EQUITY) == 1
    assert max_drawdown_duration([100, 90, 95, 97]) == 3  # never recovers
    assert max_drawdown_duration([100, 101, 102]) == 0
    dd = drawdown_series(pd.Series(EQUITY))
    np.testing.assert_allclose(dd.to_numpy(), [0, 0, 0.1, 0])
    assert calmar_ratio(0.2, 0.1) == pytest.approx(2.0)


def test_turnover_and_exposure() -> None:
    equity = [100.0, 100.0, 100.0]
    traded = [100.0, 0.0, 100.0]
    # 200 traded over 2 periods = 1 year at 2 periods per year, on 100 average equity
    assert annualized_turnover(traded, equity, periods_per_year=2) == pytest.approx(2.0)
    assert average_exposure([50.0, 100.0, 0.0], equity) == pytest.approx(0.5)
    assert time_in_market([50.0, 100.0, 0.0]) == pytest.approx(2 / 3)


def test_trade_stats() -> None:
    trades = pd.DataFrame(
        {
            "status": ["closed", "closed", "closed", "closed", "open"],
            "net_pnl": [100.0, -50.0, 30.0, -20.0, 999.0],
            "bars_held": [1, 2, 3, 4, 5],
        }
    )
    s = trade_stats(trades)
    assert s.n_trades == 4
    assert s.hit_rate == 0.5
    assert s.profit_factor == pytest.approx(130 / 70)
    assert s.avg_win == 65
    assert s.avg_loss == -35
    assert s.avg_trade == 15
    assert s.avg_bars_held == 2.5
    only_wins = trade_stats(trades.iloc[[0]])
    assert only_wins.profit_factor == math.inf
    assert trade_stats(trades.iloc[[4]]).n_trades == 0


def test_beta_alpha_recovers_linear_relation() -> None:
    rng = np.random.default_rng(1)
    bench = rng.normal(0.0005, 0.01, 1000)
    strat = 0.0002 + 2.0 * bench
    beta, alpha = beta_alpha(strat, bench, 252)
    assert beta == pytest.approx(2.0)
    assert alpha == pytest.approx(0.0002 * 252)
    idx = pd.RangeIndex(1000)
    stats = compare(
        pd.Series(strat, idx),
        pd.Series(bench, idx),
        pd.Series(np.cumprod(1 + bench), idx),
    )
    assert stats.correlation == pytest.approx(1.0)
    assert stats.beta == pytest.approx(2.0)


def test_buy_and_hold_is_equal_weight_without_rebalancing() -> None:
    prices = pd.DataFrame(
        {"A": [10.0, 20.0, 20.0], "B": [50.0, 50.0, 25.0], "C": [np.nan, 1.0, 2.0]}
    )
    eq = buy_and_hold(prices, 1_000)
    # half in A (doubles), half in B (halves at the end); C not listed at the start
    np.testing.assert_allclose(eq.to_numpy(), [1_000, 1_500, 1_250])
