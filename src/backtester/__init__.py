"""Event-driven backtesting engine for systematic trading strategies."""

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
    "CancelEvent",
    "ConfigError",
    "DataError",
    "Event",
    "EventQueue",
    "FillEvent",
    "LookAheadError",
    "MarketEvent",
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
]
