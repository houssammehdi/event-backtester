"""Compare the event engine with the vectorized fast path on every built-in strategy.

Both paths share the same execution convention (decide at the close, trade at the next
open, fixed bps slippage and commission, fractional shares), so their equity curves
should agree to floating-point precision while the fast path runs much faster.

Run:  python examples/vectorized_parity.py
"""

import time

import numpy as np

from backtester import BpsCommission, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.research import run_vectorized
from backtester.strategies import STRATEGIES


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    config = BacktestConfig(
        slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), lot_size=None
    )
    print(
        f"{'strategy':10s} {'event (s)':>10s} {'vector (s)':>11s} {'speed-up':>9s} "
        f"{'max |rel diff|':>15s}"
    )
    for name, cls in STRATEGIES.items():
        t0 = time.perf_counter()
        event = config.run(feed, cls())
        t1 = time.perf_counter()
        fast = run_vectorized(feed, cls(), slippage_bps=2.0, commission_bps=1.0)
        t2 = time.perf_counter()
        diff = np.max(np.abs(fast.equity.to_numpy() / event.equity.to_numpy() - 1.0))
        print(
            f"{name:10s} {t1 - t0:10.3f} {t2 - t1:11.4f} {(t1 - t0) / (t2 - t1):8.0f}x {diff:15.1e}"
        )


if __name__ == "__main__":
    main()
