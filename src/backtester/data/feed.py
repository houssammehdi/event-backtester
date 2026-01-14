"""Aligned multi-symbol OHLCV data and the read-only, time-pinned market view."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.errors import DataError, LookAheadError

Field = Literal["open", "high", "low", "close", "volume"]
FIELDS: tuple[Field, ...] = ("open", "high", "low", "close", "volume")
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar of a single symbol."""

    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def _readonly(array: FloatArray) -> FloatArray:
    array.flags.writeable = False
    return array


def _validate_frame(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise and validate one symbol's OHLCV frame."""
    if not isinstance(frame, pd.DataFrame):
        raise DataError(f"{symbol}: expected a DataFrame, got {type(frame).__name__}")
    df = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    missing = [f for f in FIELDS if f not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing required columns {missing}")
    df = df.loc[:, list(FIELDS)].astype("float64")
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.index))
        except (TypeError, ValueError) as exc:
            raise DataError(f"{symbol}: index must be datetime-like") from exc
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()][0]
        raise DataError(f"{symbol}: duplicate timestamp {dup}")
    df = df.sort_index()
    # Rows that are entirely NaN are treated as missing bars; partial rows are errors.
    all_nan = df.isna().all(axis=1)
    df = df.loc[~all_nan]
    values = df.to_numpy()
    if np.isnan(values).any():
        bad = df.index[np.isnan(values).any(axis=1)][0]
        raise DataError(f"{symbol}: incomplete bar at {bad}")
    o, h, lo, c, v = (values[:, k] for k in range(5))
    if not np.isfinite(values).all():
        raise DataError(f"{symbol}: non-finite values")
    if (values[:, :4] <= 0).any():
        raise DataError(f"{symbol}: prices must be strictly positive")
    if (v < 0).any():
        raise DataError(f"{symbol}: volume must be non-negative")
    tol = 1e-9
    bad_hi = h < np.maximum(o, c) * (1 - tol)
    bad_lo = lo > np.minimum(o, c) * (1 + tol)
    if bad_hi.any() or bad_lo.any():
        idx = df.index[bad_hi | bad_lo][0]
        raise DataError(f"{symbol}: inconsistent OHLC at {idx} (need low <= open,close <= high)")
    return df


class DataFeed:
    """Multi-symbol OHLCV data aligned on a common calendar.

    The calendar is the sorted union of every symbol's timestamps. When a symbol has
    no bar at a calendar timestamp (holiday, halt, not yet listed) its fields are
    ``NaN`` for that row; such a symbol cannot trade on that bar and is valued at its
    last known close.

    The feed itself exposes the full history and is meant for the engine and for
    research code. Strategies never receive the feed: they receive a
    :class:`MarketView` pinned to the current bar.
    """

    __slots__ = ("_arrays", "_has_bar", "_index", "_last_close", "_positions", "_symbols")

    def __init__(self, frames: Mapping[str, pd.DataFrame]) -> None:
        if not frames:
            raise DataError("DataFeed needs at least one symbol")
        cleaned = {str(sym): _validate_frame(str(sym), df) for sym, df in frames.items()}
        tz_set = {str(pd.DatetimeIndex(df.index).tz) for df in cleaned.values()}
        if len(tz_set) > 1:
            raise DataError(f"symbols use different time zones: {sorted(tz_set)}")
        index = cleaned[next(iter(cleaned))].index
        for df in cleaned.values():
            index = index.union(df.index)
        if len(index) == 0:
            raise DataError("DataFeed has no bars")
        self._index: pd.DatetimeIndex = pd.DatetimeIndex(index)
        self._symbols: tuple[str, ...] = tuple(cleaned)
        self._positions: dict[str, int] = {s: j for j, s in enumerate(self._symbols)}
        n, m = len(self._index), len(self._symbols)
        arrays: dict[str, FloatArray] = {f: np.full((n, m), np.nan) for f in FIELDS}
        for j, df in enumerate(cleaned.values()):
            pos = self._index.get_indexer(df.index)
            for f in FIELDS:
                arrays[f][pos, j] = df[f].to_numpy()
        self._arrays = {f: _readonly(a) for f, a in arrays.items()}
        has_bar = ~np.isnan(arrays["close"])
        has_bar.flags.writeable = False
        self._has_bar: BoolArray = has_bar
        last_close = pd.DataFrame(arrays["close"]).ffill().to_numpy(dtype=np.float64)
        self._last_close = _readonly(last_close)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> DataFeed:
        """Build a feed from a mapping ``symbol -> OHLCV DataFrame``."""
        return cls(frames)

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        *,
        date_column: str = "date",
        symbol_column: str = "symbol",
    ) -> DataFeed:
        """Build a feed from a long table with one row per ``(date, symbol)``."""
        from backtester.data.loaders import split_long_frame

        return cls(split_long_frame(frame, date_column=date_column, symbol_column=symbol_column))

    @classmethod
    def from_csv(cls, path: str | Path) -> DataFeed:
        """Load a CSV file (long format) or a directory of per-symbol CSV files."""
        from backtester.data.loaders import load_csv

        return cls(load_csv(path))

    # ------------------------------------------------------------------ properties
    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols in column order."""
        return self._symbols

    @property
    def index(self) -> pd.DatetimeIndex:
        """The aligned calendar."""
        return self._index

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        start, end = self._index[0].date(), self._index[-1].date()
        return f"DataFeed(symbols={len(self._symbols)}, bars={len(self)}, {start}..{end})"

    def array(self, field: Field) -> FloatArray:
        """Full read-only ``(bars, symbols)`` array of one field (NaN where missing)."""
        return self._arrays[field]

    @property
    def has_bar(self) -> BoolArray:
        """Read-only ``(bars, symbols)`` mask of which symbols traded on each bar."""
        return self._has_bar

    @property
    def last_close(self) -> FloatArray:
        """Read-only forward-filled closes used for valuation."""
        return self._last_close

    def frame(self, field: Field = "close") -> pd.DataFrame:
        """Full history of one field as a DataFrame (research use only)."""
        return pd.DataFrame(self._arrays[field].copy(), index=self._index, columns=self._symbols)

    def symbol_frame(self, symbol: str) -> pd.DataFrame:
        """OHLCV frame of one symbol with missing bars dropped."""
        j = self._symbol_pos(symbol)
        data = {f: self._arrays[f][:, j] for f in FIELDS}
        df = pd.DataFrame(data, index=self._index)
        return df.loc[self._has_bar[:, j]].copy()

    def _symbol_pos(self, symbol: str) -> int:
        try:
            return self._positions[symbol]
        except KeyError:
            raise KeyError(f"unknown symbol {symbol!r}") from None

    # ------------------------------------------------------------------ slicing & views
    def slice(self, start: int, stop: int) -> DataFeed:
        """Return a new feed restricted to calendar positions ``[start, stop)``."""
        if not 0 <= start < stop <= len(self):
            raise DataError(f"invalid slice [{start}, {stop}) for feed of length {len(self)}")
        idx = self._index[start:stop]
        frames = {}
        for j, sym in enumerate(self._symbols):
            data = {f: self._arrays[f][start:stop, j] for f in FIELDS}
            frames[sym] = pd.DataFrame(data, index=idx)
        # Symbols with no bars inside the slice are dropped.
        frames = {s: df for s, df in frames.items() if df["close"].notna().any()}
        return DataFeed(frames)

    def select(self, symbols: Sequence[str]) -> DataFeed:
        """Return a new feed containing only ``symbols``."""
        return DataFeed({s: self.symbol_frame(s) for s in symbols})

    def view(self, position: int) -> MarketView:
        """Return the read-only view of the market as of calendar ``position``."""
        if not 0 <= position < len(self):
            raise IndexError(f"position {position} outside feed of length {len(self)}")
        return MarketView(self, position)

    def __iter__(self) -> Iterator[MarketView]:
        for i in range(len(self)):
            yield MarketView(self, i)


class MarketView:
    """What a strategy is allowed to know at one point in time.

    A view is pinned to a single calendar position ``t`` and is immutable. Every
    accessor returns data for bars ``<= t`` only; arrays are read-only NumPy views that
    physically end at ``t`` (so ``window(...)[t + 1]`` is an ``IndexError``), and any
    explicit request for a later timestamp raises :class:`~backtester.LookAheadError`.
    """

    __slots__ = ("_feed", "_pos")
    _feed: DataFeed
    _pos: int

    def __init__(self, feed: DataFeed, position: int) -> None:
        object.__setattr__(self, "_feed", feed)
        object.__setattr__(self, "_pos", position)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MarketView is read-only")

    def __repr__(self) -> str:
        return f"MarketView(t={self.timestamp}, position={self._pos})"

    # ------------------------------------------------------------------ basics
    @property
    def timestamp(self) -> pd.Timestamp:
        """Timestamp of the current (just closed) bar."""
        return self._feed.index[self._pos]

    @property
    def position(self) -> int:
        """Calendar position of the current bar (number of prior bars)."""
        return self._pos

    @property
    def symbols(self) -> tuple[str, ...]:
        """Symbols of the universe, in column order."""
        return self._feed.symbols

    @property
    def index(self) -> pd.DatetimeIndex:
        """Calendar up to and including the current bar."""
        return self._feed.index[: self._pos + 1]

    def _check_time(self, timestamp: pd.Timestamp) -> int:
        ts = pd.Timestamp(timestamp)
        if ts > self.timestamp:
            raise LookAheadError(
                f"requested {ts} but the current bar is {self.timestamp}: that is the future"
            )
        pos = int(self._feed.index.searchsorted(ts, side="right")) - 1
        if pos < 0:
            raise KeyError(f"{ts} is before the start of the data")
        return pos

    # ------------------------------------------------------------------ current bar
    def has_bar(self, symbol: str) -> bool:
        """Whether ``symbol`` printed a bar at the current timestamp."""
        return bool(self._feed.has_bar[self._pos, self._feed._symbol_pos(symbol)])

    def bar(self, symbol: str) -> Bar | None:
        """The current bar of ``symbol``, or ``None`` if it did not trade."""
        return self._bar_at_pos(self._pos, symbol)

    def bar_at(self, timestamp: pd.Timestamp, symbol: str) -> Bar | None:
        """The bar of ``symbol`` at or immediately before ``timestamp``.

        Raises:
            LookAheadError: if ``timestamp`` is after the current bar.
        """
        return self._bar_at_pos(self._check_time(timestamp), symbol)

    def _bar_at_pos(self, pos: int, symbol: str) -> Bar | None:
        j = self._feed._symbol_pos(symbol)
        if not self._feed.has_bar[pos, j]:
            return None
        a = self._feed.array
        return Bar(
            timestamp=self._feed.index[pos],
            symbol=symbol,
            open=float(a("open")[pos, j]),
            high=float(a("high")[pos, j]),
            low=float(a("low")[pos, j]),
            close=float(a("close")[pos, j]),
            volume=float(a("volume")[pos, j]),
        )

    def price(self, symbol: str) -> float:
        """Last known close of ``symbol`` (NaN if it has never traded)."""
        return float(self._feed.last_close[self._pos, self._feed._symbol_pos(symbol)])

    def prices(self) -> dict[str, float]:
        """Last known close of every symbol."""
        row = self._feed.last_close[self._pos]
        return {s: float(p) for s, p in zip(self._feed.symbols, row, strict=True)}

    # ------------------------------------------------------------------ history
    def window(self, field: Field = "close", lookback: int | None = None) -> FloatArray:
        """Read-only ``(n, symbols)`` array of the last ``lookback`` bars (all if None).

        Rows end at the current bar; missing bars are ``NaN``.
        """
        stop = self._pos + 1
        start = 0 if lookback is None else max(stop - _positive(lookback), 0)
        return self._feed.array(field)[start:stop]

    def filled_closes(self, lookback: int | None = None) -> FloatArray:
        """Like :meth:`window` for closes, but forward-filled across missing bars."""
        stop = self._pos + 1
        start = 0 if lookback is None else max(stop - _positive(lookback), 0)
        return self._feed.last_close[start:stop]

    def series(
        self, symbol: str, field: Field = "close", lookback: int | None = None
    ) -> FloatArray:
        """Last ``lookback`` *actual* bars of one symbol (missing bars skipped)."""
        j = self._feed._symbol_pos(symbol)
        col = self._feed.array(field)[: self._pos + 1, j]
        mask = self._feed.has_bar[: self._pos + 1, j]
        values = col[mask]
        if lookback is not None:
            values = values[-_positive(lookback) :]
        return values

    def history(self, field: Field = "close", lookback: int | None = None) -> pd.DataFrame:
        """Copy of the recent history of ``field`` as a DataFrame indexed by timestamp."""
        data = self.window(field, lookback)
        idx = self._feed.index[self._pos + 1 - len(data) : self._pos + 1]
        return pd.DataFrame(data.copy(), index=idx, columns=self._feed.symbols)

    def __len__(self) -> int:
        return self._pos + 1


def _positive(value: int) -> int:
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"lookback must be a positive integer, got {value}")
    return int(value)
