"""Market data: aligned feeds, read-only views, CSV I/O and synthetic generation."""

from backtester.data.feed import FIELDS, Bar, DataFeed, Field, MarketView
from backtester.data.loaders import load_csv, save_csv, split_long_frame
from backtester.data.synthetic import (
    ASSET_CLASSES,
    Regime,
    SyntheticConfig,
    SyntheticMarket,
    generate_market,
    generate_multi_asset,
    generate_ohlcv,
)

__all__ = [
    "ASSET_CLASSES",
    "FIELDS",
    "Bar",
    "DataFeed",
    "Field",
    "MarketView",
    "Regime",
    "SyntheticConfig",
    "SyntheticMarket",
    "generate_market",
    "generate_multi_asset",
    "generate_ohlcv",
    "load_csv",
    "save_csv",
    "split_long_frame",
]
