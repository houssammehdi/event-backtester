"""Event-driven backtesting engine for systematic trading strategies.

The public API is re-exported here::

    from backtester import DataFeed, Engine, SimulatedBroker, RiskLimits, generate_ohlcv
    from backtester.strategies import TimeSeriesMomentum

    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    result = Engine(feed, TimeSeriesMomentum(), initial_cash=1_000_000).run()
    print(result.report())
"""

from backtester.data import (
    Bar,
    DataFeed,
    MarketView,
    generate_market,
    generate_multi_asset,
    generate_ohlcv,
    load_csv,
)
from backtester.engine import BacktestResult, Engine
from backtester.errors import (
    AccountingError,
    BacktesterError,
    ConfigError,
    DataError,
    LookAheadError,
    OrderError,
)
from backtester.events import (
    CancelEvent,
    Event,
    EventQueue,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    TargetEvent,
)
from backtester.execution import (
    BpsCommission,
    FixedBpsSlippage,
    IntrabarPath,
    NoCommission,
    NoSlippage,
    PerShareCommission,
    SimulatedBroker,
    SquareRootImpactSlippage,
)
from backtester.orders import Order, OrderStatus, OrderType, Side, TimeInForce
from backtester.portfolio import Portfolio, Position
from backtester.risk import RiskLimits, RiskManager
from backtester.strategy import Strategy, StrategyContext, VectorizedStrategy
from backtester.strategy.rebalancing import TargetWeightStrategy

__version__ = "0.1.0"

__all__ = [
    "AccountingError",
    "BacktestResult",
    "BacktesterError",
    "Bar",
    "BpsCommission",
    "CancelEvent",
    "ConfigError",
    "DataError",
    "DataFeed",
    "Engine",
    "Event",
    "EventQueue",
    "FillEvent",
    "FixedBpsSlippage",
    "IntrabarPath",
    "LookAheadError",
    "MarketEvent",
    "MarketView",
    "NoCommission",
    "NoSlippage",
    "Order",
    "OrderError",
    "OrderEvent",
    "OrderStatus",
    "OrderType",
    "PerShareCommission",
    "Portfolio",
    "Position",
    "RiskLimits",
    "RiskManager",
    "Side",
    "SignalEvent",
    "SimulatedBroker",
    "SquareRootImpactSlippage",
    "Strategy",
    "StrategyContext",
    "TargetEvent",
    "TargetWeightStrategy",
    "TimeInForce",
    "VectorizedStrategy",
    "__version__",
    "generate_market",
    "generate_multi_asset",
    "generate_ohlcv",
    "load_csv",
]
