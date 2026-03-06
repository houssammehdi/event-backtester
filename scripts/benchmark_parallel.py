"""Benchmark parallel grid search: wall time for several ``n_jobs`` on the same grid.

The grid is TimeSeriesMomentum over 18 configurations on 10 synthetic symbols and 10
years. Each ``n_jobs`` setting runs ``--repeat`` times (median reported), and every
parallel result is checked against the serial one. Worker start-up (a fresh
interpreter importing NumPy and pandas, then receiving the feed) is part of the
measured time, as it is for a user.

Run:  python scripts/benchmark_parallel.py [--repeat 3] [--jobs 1 2 4]
"""

from __future__ import annotations

import argparse
import statistics
import time

import pandas as pd

from backtester import BpsCommission, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.research import grid_search
from backtester.research.parallel import available_cpus
from backtester.strategies import TimeSeriesMomentum

CONFIG = BacktestConfig(
    slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), max_participation=0.1
)
GRID = {"lookback": [63, 126, 252], "vol_lookback": [21, 63, 126], "target_vol": [0.1, 0.2]}


def main() -> None:
    """Time the grid search for each ``--jobs`` value and check the results agree."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=3, help="runs per setting")
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    feed = DataFeed(generate_ohlcv(n_symbols=10, years=10, seed=42))
    n_configs = len(GRID["lookback"]) * len(GRID["vol_lookback"]) * len(GRID["target_vol"])
    print(f"{n_configs} configurations, {len(feed)} bars x {len(feed.symbols)} symbols")
    print(f"CPUs available: {available_cpus()}")
    reference: pd.DataFrame | None = None
    baseline = 0.0
    for jobs in args.jobs:
        times = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            search = grid_search(feed, TimeSeriesMomentum, GRID, config=CONFIG, n_jobs=jobs)
            times.append(time.perf_counter() - start)
        if reference is None:
            reference, baseline = search.table, statistics.median(times)
        pd.testing.assert_frame_equal(search.table, reference, check_exact=True)
        median = statistics.median(times)
        print(f"n_jobs={jobs:<3d} median {median:6.2f} s   x{baseline / median:4.2f}")
    print(f"(speed-ups relative to n_jobs={args.jobs[0]}; results identical across settings)")


if __name__ == "__main__":
    main()
