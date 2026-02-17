"""Compare portfolio-construction methods on a multi-asset universe, honestly.

EW, inverse volatility, minimum variance, equal risk contribution and HRP, rebalanced
monthly on a 252-bar Ledoit-Wolf covariance, with 2 bps slippage and 1 bps commission,
on a synthetic universe of 6 equities, 4 bonds and 4 commodities (each asset class has
its own market factor; see ``generate_multi_asset``).

Every allocation uses trailing data only, so each backtest is out of sample in the
sense that nothing is fitted to the evaluation period. The remaining question is
selection: having looked at five methods, is the best one really better than 1/N?

1. One market (seed 42), event engine: performance with bootstrap intervals.
2. Data snooping: Hansen's SPA and White's Reality Check with EW as the benchmark,
   and the probability of backtest overfitting of "pick the best method". SPA tests
   mean returns, and the risk-based portfolios run at a fraction of EW's volatility,
   so the risk-adjusted test first scales each method to EW's volatility (ex post:
   a device for comparing Sharpe ratios, not a tradable strategy).
3. Walk-forward selection of the method (3 years in-sample, 1 year out).
4. The same comparison on 20 independently seeded markets (vectorized fast path,
   same accounting).

Run:  python examples/allocation_study.py            (about 1-2 minutes)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from backtester import BpsCommission, DataFeed, FixedBpsSlippage, generate_multi_asset
from backtester.analytics.report import num, pct, table
from backtester.config import BacktestConfig
from backtester.research import run_vectorized, walk_forward, walk_forward_windows
from backtester.strategies import AllocationStrategy
from backtester.validation import (
    bootstrap_statistics,
    probability_of_backtest_overfitting,
    superior_predictive_ability,
    validate_returns,
)

METHODS = ("ew", "iv", "minvar", "erc", "hrp")
SEEDS = range(20)
CONFIG = BacktestConfig(
    slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), max_participation=0.1
)


def one_market(feed: DataFeed) -> pd.DataFrame:
    returns = {}
    rows = []
    for method in METHODS:
        result = CONFIG.run(feed, AllocationStrategy(method))
        m = result.metrics()
        v = validate_returns(result.returns, n_samples=2000)
        sr = v.bootstrap["sharpe"]
        dd = v.bootstrap["max_drawdown"]
        rows.append(
            (
                method,
                pct(m.cagr),
                pct(m.annual_volatility),
                f"{num(sr.estimate)} [{num(sr.lower)}, {num(sr.upper)}]",
                f"{pct(dd.estimate, 1)} [{pct(dd.lower, 1)}, {pct(dd.upper, 1)}]",
                pct(m.turnover, 0),
                num(m.total_commission + m.total_slippage, 0),
            )
        )
        returns[method] = result.returns
    header = ("method", "CAGR", "vol", "Sharpe [95% CI]", "max DD [95% CI]", "turnover", "costs")
    print(table(rows, header=header))
    return pd.DataFrame(returns)


def at_ew_risk(returns: pd.DataFrame) -> pd.DataFrame:
    """Scale every method to EW's (ex-post) volatility, so mean returns compare Sharpe."""
    return returns * (returns["ew"].std() / returns.std())


def snooping(returns: pd.DataFrame) -> None:
    pbo = probability_of_backtest_overfitting(returns, n_splits=16)
    best = returns.apply(lambda r: r.mean() / r.std()).idxmax()
    raw = superior_predictive_ability(returns.drop(columns="ew"), returns["ew"], n_samples=2000)
    scaled = at_ew_risk(returns)
    matched = superior_predictive_ability(scaled.drop(columns="ew"), scaled["ew"], n_samples=2000)
    rows = [
        ("Best full-sample Sharpe", best),
        ("PBO of picking the best method", pct(pbo.pbo, 1)),
        ("SPA p-value, raw returns", num(raw.pvalue_consistent, 3)),
        ("SPA p-value, at EW's volatility", num(matched.pvalue_consistent, 3)),
        ("Reality Check p-value, at EW's volatility", num(matched.reality_check_pvalue, 3)),
    ]
    print(table(rows, header=("Selection among the five methods (H0: none beats EW)", "")))
    diff = scaled[best] - scaled["ew"]
    ci = bootstrap_statistics(diff, {"mean": lambda p: p.mean(axis=1) * 252}, n_samples=2000)
    iv = ci["mean"]
    print(
        f"At EW's volatility, {best} minus EW earns {pct(iv.estimate)} a year "
        f"[95% CI {pct(iv.lower)}, {pct(iv.upper)}]"
    )


def walk_forward_selection(feed: DataFeed) -> None:
    windows = walk_forward_windows(len(feed), train_size=756, test_size=252, start=252)
    wf = walk_forward(feed, AllocationStrategy, {"method": list(METHODS)}, windows, config=CONFIG)
    chosen = ", ".join(str(m) for m in wf.windows["param_method"])
    ew = walk_forward(feed, AllocationStrategy, {"method": ["ew"]}, windows, config=CONFIG)
    rows = []
    for label, result in (("selected in-sample", wf), ("always EW", ew)):
        v = result.validate(n_samples=2000)
        sr = v.bootstrap["sharpe"]
        rows.append(
            (
                label,
                num(result.metrics()["sharpe"]),
                f"[{num(sr.lower)}, {num(sr.upper)}]",
                pct(result.metrics()["cagr"]),
                pct(result.metrics()["max_drawdown"]),
            )
        )
    print(f"Methods chosen per window: {chosen}")
    header = ("Walk-forward OOS", "Sharpe", "95% CI", "CAGR", "max DD")
    print(table(rows, header=header))


def seed_study() -> None:
    sharpe: dict[str, list[float]] = {m: [] for m in METHODS}
    rejections = 0
    for seed in SEEDS:
        feed = DataFeed(generate_multi_asset(years=10, seed=seed))
        rets = {}
        for method in METHODS:
            fast = run_vectorized(
                feed, AllocationStrategy(method), slippage_bps=2.0, commission_bps=1.0
            )
            sharpe[method].append(fast.summary()["sharpe"])
            rets[method] = fast.returns
        frame = at_ew_risk(pd.DataFrame(rets))
        spa = superior_predictive_ability(frame.drop(columns="ew"), frame["ew"], n_samples=1000)
        rejections += spa.pvalue_consistent < 0.05
    ew = np.array(sharpe["ew"])
    rows = []
    for method in METHODS:
        s = np.array(sharpe[method])
        rows.append(
            (
                method,
                num(s.mean()),
                num(s.std(ddof=1)),
                num(s.min()),
                num(s.max()),
                "" if method == "ew" else pct(float(np.mean(s > ew)), 0),
            )
        )
    print(f"Sharpe ratios over {len(SEEDS)} seeds (10 years each, same costs)")
    print(table(rows, header=("method", "mean", "std", "min", "max", "beats EW")))
    print(
        f"SPA at EW's volatility rejects 'nothing beats EW' at 5% in {rejections} of "
        f"{len(SEEDS)} markets."
    )


def main() -> None:
    t0 = time.perf_counter()
    feed = DataFeed(generate_multi_asset(years=10, seed=42))
    print(f"Universe: {len(feed.symbols)} assets, {len(feed)} bars, seed 42\n")
    returns = one_market(feed)
    print()
    snooping(returns)
    print()
    walk_forward_selection(feed)
    print()
    seed_study()
    print(f"\nTotal time {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
