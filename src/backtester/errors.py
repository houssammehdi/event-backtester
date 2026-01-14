"""Exception hierarchy used across the package."""

from __future__ import annotations


class BacktesterError(Exception):
    """Base class for all errors raised by :mod:`backtester`."""


class DataError(BacktesterError, ValueError):
    """Raised when market data is malformed or inconsistent."""


class LookAheadError(BacktesterError, IndexError):
    """Raised when code tries to read market data from the future.

    Strategies receive a :class:`~backtester.data.MarketView` that is pinned to the
    current bar. Any attempt to access a later bar raises this error instead of
    silently returning data the strategy could not have known.
    """


class OrderError(BacktesterError, ValueError):
    """Raised when an order is malformed (bad quantity, missing limit price, ...)."""


class ConfigError(BacktesterError, ValueError):
    """Raised for invalid configuration values."""


class AccountingError(BacktesterError, RuntimeError):
    """Raised when a portfolio accounting identity is violated (a bug, never data)."""
