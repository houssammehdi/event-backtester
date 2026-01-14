"""CSV loading and saving of OHLCV data."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from backtester.errors import DataError


def split_long_frame(
    frame: pd.DataFrame,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> dict[str, pd.DataFrame]:
    """Split a long ``(date, symbol, open, high, low, close, volume)`` table per symbol.

    Column names are matched case-insensitively. Extra columns are ignored by the feed.
    """
    df = frame.rename(columns={c: str(c).strip().lower() for c in frame.columns})
    date_col, sym_col = date_column.lower(), symbol_column.lower()
    for col in (date_col, sym_col):
        if col not in df.columns:
            raise DataError(f"long frame is missing the {col!r} column")
    df[date_col] = pd.to_datetime(df[date_col])
    out: dict[str, pd.DataFrame] = {}
    for symbol, group in df.groupby(sym_col, sort=True):
        out[str(symbol)] = group.drop(columns=[sym_col]).set_index(date_col).sort_index()
    if not out:
        raise DataError("long frame contains no rows")
    return out


def _read_single(path: Path, *, date_column: str, symbol_column: str) -> dict[str, pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    columns = {str(c).strip().lower() for c in df.columns}
    if symbol_column.lower() in columns:
        return split_long_frame(df, date_column=date_column, symbol_column=symbol_column)
    lowered = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if date_column.lower() not in lowered.columns:
        raise DataError(f"{path}: missing the {date_column!r} column")
    lowered[date_column.lower()] = pd.to_datetime(lowered[date_column.lower()])
    return {path.stem: lowered.set_index(date_column.lower()).sort_index()}


def load_csv(
    path: str | Path,
    *,
    date_column: str = "date",
    symbol_column: str = "symbol",
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data from CSV.

    ``path`` may be:

    * a single file in *long* format with a ``symbol`` column, or
    * a single file without a ``symbol`` column (the file stem becomes the symbol), or
    * a directory, in which every ``*.csv`` file is loaded as above and merged.

    Returns:
        Mapping ``symbol -> DataFrame`` indexed by timestamp, ready for
        :class:`~backtester.data.DataFeed`.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise DataError(f"no CSV files found in {p}")
        merged: dict[str, pd.DataFrame] = {}
        for f in files:
            for symbol, df in _read_single(
                f, date_column=date_column, symbol_column=symbol_column
            ).items():
                if symbol in merged:
                    raise DataError(f"symbol {symbol!r} appears in more than one file")
                merged[symbol] = df
        return merged
    if not p.exists():
        raise DataError(f"{p} does not exist")
    return _read_single(p, date_column=date_column, symbol_column=symbol_column)


def save_csv(frames: Mapping[str, pd.DataFrame], path: str | Path) -> Path:
    """Write ``symbol -> OHLCV frame`` data to a single long-format CSV file."""
    parts = []
    for symbol, df in frames.items():
        part = df.copy()
        part.index.name = "date"
        part = part.reset_index()
        part.insert(1, "symbol", symbol)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"], kind="stable")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, date_format="%Y-%m-%d")
    return target
