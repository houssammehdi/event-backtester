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
from backtester.orders import Order, OrderStatus, OrderType, Side, TimeInForce

__version__ = "0.1.0"

__all__ = [
    "AccountingError",
    "BacktesterError",
    "Bar",
    "CancelEvent",
    "ConfigError",
    "DataError",
    "DataFeed",
    "Event",
    "EventQueue",
    "FillEvent",
    "LookAheadError",
    "MarketEvent",
    "MarketView",
    "Order",
    "OrderError",
    "OrderEvent",
    "OrderStatus",
    "OrderType",
    "Side",
    "SignalEvent",
    "TargetEvent",
    "TimeInForce",
    "__version__",
    "generate_market",
    "generate_ohlcv",
    "load_csv",
]
