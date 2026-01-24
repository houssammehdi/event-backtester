"""Run every built-in strategy on the same synthetic market and tabulate the results.

Run:  python examples/compare_strategies.py
"""

from backtester import BpsCommission, DataFeed, FixedBpsSlippage, RiskLimits, generate_ohlcv
from backtester.analytics.report import num, pct, table
from backtester.config import BacktestConfig
from backtester.strategies import STRATEGIES


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    config = BacktestConfig(
        slippage=FixedBpsSlippage(2.0),
        commission=BpsCommission(1.0),
        max_participation=0.1,
        limits=RiskLimits(max_gross_leverage=2.0),
    )
    rows = []
    benchmark = None
    for name, cls in STRATEGIES.items():
        m = config.run(feed, cls()).metrics()
        benchmark = m.benchmark
        rows.append(
            (
                name,
                pct(m.cagr),
                pct(m.annual_volatility),
                num(m.sharpe),
                pct(m.max_drawdown),
                pct(m.turnover, 0),
                str(m.trades.n_trades),
            )
        )
    if benchmark is not None:
        rows.append(
            (
                "buy & hold",
                pct(benchmark.cagr),
                pct(benchmark.volatility),
                num(benchmark.sharpe),
                pct(benchmark.max_drawdown),
                "0%",
                "-",
            )
        )
    header = ("strategy", "CAGR", "vol", "Sharpe", "max DD", "turnover", "trades")
    print(table(rows, header=header))


if __name__ == "__main__":
    main()
