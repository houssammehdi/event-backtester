"""How much of a single backtest is luck? Re-run every strategy on 20 synthetic markets.

A single seed is one draw of history. This study uses the vectorized fast path (same
accounting as the event engine, much faster) to backtest each built-in strategy on 20
independently seeded 10-year markets and summarises the distribution of Sharpe ratios
against equal-weight buy-and-hold on the same data.

Run:  python examples/seed_study.py
"""

import numpy as np

from backtester import DataFeed, generate_ohlcv
from backtester.analytics import buy_and_hold, sharpe_ratio
from backtester.analytics.report import num, pct, table
from backtester.research import run_vectorized
from backtester.strategies import STRATEGIES

SEEDS = range(20)


def main() -> None:
    sharpes: dict[str, list[float]] = {name: [] for name in STRATEGIES}
    bench: list[float] = []
    for seed in SEEDS:
        feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=seed))
        for name, cls in STRATEGIES.items():
            result = run_vectorized(feed, cls(), slippage_bps=2.0, commission_bps=1.0)
            sharpes[name].append(result.summary()["sharpe"])
        bh = buy_and_hold(feed.frame("close"))
        bench.append(sharpe_ratio(bh.pct_change().iloc[1:]))
    b = np.array(bench)
    rows = []
    for name, values in sharpes.items():
        s = np.array(values)
        rows.append(
            (
                name,
                num(s.mean()),
                num(s.std(ddof=1)),
                num(s.min()),
                num(s.max()),
                pct(float(np.mean(s > b)), 0),
            )
        )
    rows.append(("buy & hold", num(b.mean()), num(b.std(ddof=1)), num(b.min()), num(b.max()), ""))
    header = ("strategy", "mean Sharpe", "std", "min", "max", "beats B&H")
    print(f"Sharpe ratios across {len(SEEDS)} seeds (5 symbols, 10 years, 2 bps + 1 bps costs)")
    print(table(rows, header=header))


if __name__ == "__main__":
    main()
