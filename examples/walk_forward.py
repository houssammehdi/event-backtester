"""Walk-forward optimisation of cross-sectional momentum on synthetic data.

Rolling 3-year in-sample fits pick the look-back and portfolio size; each choice is then
traded on the following, unseen year. The stitched out-of-sample curve is the honest
estimate of performance - the in-sample optimum is not.

Run:  python examples/walk_forward.py
"""

from backtester import BpsCommission, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.research import walk_forward, walk_forward_windows
from backtester.strategies import CrossSectionalMomentum


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=8, years=12, seed=7))
    config = BacktestConfig(
        slippage=FixedBpsSlippage(2.0), commission=BpsCommission(1.0), max_participation=0.1
    )
    windows = walk_forward_windows(len(feed), train_size=756, test_size=252, start=252)
    grid = {"lookback": [63, 126, 252], "top_k": [2, 4]}
    wf = walk_forward(feed, CrossSectionalMomentum, grid, windows, config=config)
    columns = [
        "test_start",
        "test_end",
        "param_lookback",
        "param_top_k",
        "is_sharpe",
        "is_dsr",
        "oos_sharpe",
        "oos_return",
    ]
    print(wf.windows[columns].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print()
    for key, value in wf.metrics().items():
        print(f"{key:>24s}: {value:.3f}")
    print()
    print(wf.note())


if __name__ == "__main__":
    main()
