"""Event-driven backtesting engine for systematic trading strategies."""

from backtester.data import Bar, DataFeed, MarketView, generate_market, generate_ohlcv, load_csv
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
    NoCommission,
    NoSlippage,
    PerShareCommission,
    SimulatedBroker,
    SquareRootImpactSlippage,
)
from backtester.orders import Order, OrderStatus, OrderType, Side, TimeInForce

__version__ = "0.1.0"

__all__ = [
    "AccountingError",
    "BacktesterError",
    "Bar",
    "BpsCommission",
    "CancelEvent",
    "ConfigError",
    "DataError",
    "DataFeed",
    "Event",
    "EventQueue",
    "FillEvent",
    "FixedBpsSlippage",
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
    "Side",
    "SignalEvent",
    "SimulatedBroker",
    "SquareRootImpactSlippage",
    "TargetEvent",
    "TimeInForce",
    "__version__",
    "generate_market",
    "generate_ohlcv",
    "load_csv",
]
