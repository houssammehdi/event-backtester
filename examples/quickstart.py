"""Backtest time-series momentum on seeded synthetic data and print the report.

Run:  python examples/quickstart.py
"""

from backtester import (
    BpsCommission,
    DataFeed,
    Engine,
    FixedBpsSlippage,
    RiskLimits,
    SimulatedBroker,
    generate_ohlcv,
)
from backtester.strategies import TimeSeriesMomentum


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    broker = SimulatedBroker(
        slippage=FixedBpsSlippage(2.0),
        commission=BpsCommission(1.0),
        max_participation=0.10,
    )
    engine = Engine(
        feed,
        TimeSeriesMomentum(lookback=252, target_vol=0.15),
        initial_cash=1_000_000,
        broker=broker,
        risk=RiskLimits(max_gross_leverage=2.0, max_drawdown=0.35),
        check_invariants=True,
    )
    result = engine.run()
    print(result.report())


if __name__ == "__main__":
    main()
