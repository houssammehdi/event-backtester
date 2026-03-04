"""Benchmark the event engine: median wall time and bars per second per scenario.

Each scenario runs ``--repeat`` times and the median wall time of the backtest is
reported (building the data feed is not timed). Timings depend on the machine and its
load, so compare numbers from one machine and one session only. Scenarios that need
features the imported package lacks are skipped, which lets the same script measure
an older checkout::

    PYTHONPATH=/path/to/old/checkout/src python scripts/benchmark_engine.py

Run:  python scripts/benchmark_engine.py [--repeat 5] [--scenario NAME ...]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import backtester
from backtester import BpsCommission, DataFeed, FixedBpsSlippage, Strategy, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.strategies import SmaCrossover, TimeSeriesMomentum

CONFIG = BacktestConfig(
    slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), max_participation=0.1
)
Scenario = tuple[DataFeed, Callable[[], Strategy]]


def tsmom_monthly() -> Scenario:
    """The README command: time-series momentum, 5 symbols, monthly targets."""
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    return feed, TimeSeriesMomentum


def sma_daily() -> Scenario:
    """SMA crossover on 20 symbols, re-targeted every bar (many small orders)."""
    feed = DataFeed(generate_ohlcv(n_symbols=20, years=10, seed=42))
    return feed, lambda: SmaCrossover(fast=20, slow=100, rebalance="daily")


def brackets() -> Scenario:
    """Stop entries with bracket exits (examples/intrabar_sensitivity.py, tight bracket)."""
    from backtester import StrategyContext

    if not hasattr(StrategyContext, "bracket"):
        raise ImportError("this version has no bracket orders")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from intrabar_sensitivity import BracketBreakout

    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    return feed, lambda: BracketBreakout(stop_atr=0.5, target_atr=0.75)


def allocation_hrp() -> Scenario:
    """Monthly HRP allocation over the 14-asset multi-asset universe."""
    from backtester import generate_multi_asset
    from backtester.strategies import AllocationStrategy

    feed = DataFeed(generate_multi_asset(years=10, seed=42))
    return feed, lambda: AllocationStrategy("hrp")


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "tsmom_monthly_5": tsmom_monthly,
    "sma_daily_20": sma_daily,
    "brackets_5": brackets,
    "allocation_hrp_14": allocation_hrp,
}


def main() -> None:
    """Run the selected scenarios and print one line per scenario."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeat", type=int, default=5, help="runs per scenario")
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    args = parser.parse_args()
    print(f"backtester {backtester.__version__} from {Path(backtester.__file__).parent}")
    header = ("scenario", "bars", "syms", "median s", "bars/s", "final equity")
    print(f"{header[0]:18s} {header[1]:>5s} {header[2]:>4s} {header[3]:>9s} ", end="")
    print(f"{header[4]:>7s} {header[5]:>16s}")
    for name in args.scenario or list(SCENARIOS):
        try:
            feed, factory = SCENARIOS[name]()
        except ImportError as exc:
            print(f"{name:18s} skipped: {exc}")
            continue
        times = []
        final = float("nan")
        for _ in range(args.repeat):
            strategy = factory()
            start = time.perf_counter()
            result = CONFIG.run(feed, strategy)
            times.append(time.perf_counter() - start)
            final = result.final_equity
        median = statistics.median(times)
        print(
            f"{name:18s} {len(feed):5d} {len(feed.symbols):4d} {median:9.3f} "
            f"{len(feed) / median:7.0f} {final:16.6f}"
        )


if __name__ == "__main__":
    main()
