from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester import DataError, DataFeed, LookAheadError, load_csv
from backtester.data import save_csv
from tests.conftest import bars, flat_bars


def two_symbol_feed() -> DataFeed:
    a = bars([(10, 11, 9, 10.5, 100), (10.5, 12, 10, 11, 100), (11, 11, 10, 10, 100)])
    b = bars([(20, 21, 19, 20, 50), (20, 22, 19, 21, 50)], start="2024-01-02")
    return DataFeed({"A": a, "B": b})


def test_calendar_is_union_and_missing_bars_are_nan() -> None:
    feed = two_symbol_feed()
    assert list(feed.index) == list(pd.bdate_range("2024-01-01", periods=3))
    close = feed.array("close")
    assert np.isnan(close[0, 1])
    assert feed.has_bar.tolist() == [[True, False], [True, True], [True, True]]
    # valuation price is forward-filled but never back-filled
    assert np.isnan(feed.last_close[0, 1])
    assert feed.last_close[2, 1] == 21


def test_forward_filled_valuation_across_holes() -> None:
    a = bars([(10, 10, 10, 10, 1)] * 4)
    b = bars([(5, 5, 5, 5, 1)] * 4)
    b.iloc[2] = np.nan  # all-NaN row = missing bar
    feed = DataFeed({"A": a, "B": b})
    assert not feed.has_bar[2, 1]
    assert feed.last_close[2, 1] == 5
    view = feed.view(2)
    assert view.bar("B") is None
    assert not view.has_bar("B")
    assert view.price("B") == 5
    assert len(view.series("B")) == 2  # missing bar skipped


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda df: df.drop(columns="volume"), "missing required columns"),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate timestamp"),
        (lambda df: df.assign(high=df["low"] - 1), "inconsistent OHLC"),
        (lambda df: df.assign(close=-df["close"]), "strictly positive"),
        (lambda df: df.assign(volume=-1.0), "non-negative"),
        (lambda df: df.assign(open=[np.nan, 10.0, 10.0]), "incomplete bar"),
    ],
)
def test_validation_errors(mutate, message: str) -> None:
    df = bars([(10, 11, 9, 10, 1)] * 3)
    with pytest.raises(DataError, match=message):
        DataFeed({"X": mutate(df)})


def test_empty_feed_is_rejected() -> None:
    with pytest.raises(DataError):
        DataFeed({})


def test_unsorted_input_is_sorted() -> None:
    df = bars([(10, 11, 9, 10, 1), (11, 12, 10, 11, 1)])
    feed = DataFeed({"X": df.iloc[::-1]})
    assert feed.index.is_monotonic_increasing
    assert feed.array("close")[0, 0] == 10


class TestMarketViewEnforcesNoLookAhead:
    def test_window_physically_ends_at_current_bar(self) -> None:
        feed = DataFeed({"X": flat_bars(10)})
        view = feed.view(4)
        window = view.window("close")
        assert window.shape == (5, 1)
        with pytest.raises(IndexError):
            _ = window[5]
        assert view.window("close", lookback=3).shape == (3, 1)
        assert view.history("close").index[-1] == view.timestamp
        assert len(view) == 5

    def test_arrays_are_read_only(self) -> None:
        feed = DataFeed({"X": flat_bars(10)})
        view = feed.view(4)
        with pytest.raises(ValueError, match="read-only"):
            view.window("close")[0, 0] = 1.0
        with pytest.raises(ValueError, match="read-only"):
            feed.array("close")[9, 0] = 1.0

    def test_future_timestamp_raises(self) -> None:
        feed = DataFeed({"X": flat_bars(10)})
        view = feed.view(4)
        assert view.bar_at(feed.index[2], "X") is not None
        with pytest.raises(LookAheadError):
            view.bar_at(feed.index[5], "X")

    def test_view_is_immutable(self) -> None:
        view = DataFeed({"X": flat_bars(3)}).view(0)
        with pytest.raises(AttributeError):
            view._pos = 2  # type: ignore[misc]

    def test_history_mutation_does_not_leak(self) -> None:
        feed = DataFeed({"X": flat_bars(5)})
        hist = feed.view(3).history()
        hist.iloc[0, 0] = -1.0
        assert feed.array("close")[0, 0] == 100.0

    def test_invalid_lookback(self) -> None:
        view = DataFeed({"X": flat_bars(5)}).view(3)
        with pytest.raises(ValueError, match="lookback"):
            view.window("close", lookback=0)


def test_slice_and_select() -> None:
    feed = two_symbol_feed()
    head = feed.slice(0, 1)
    assert head.symbols == ("A",)  # B has no bars in the slice
    assert len(head) == 1
    assert feed.select(["B"]).symbols == ("B",)
    with pytest.raises(DataError):
        feed.slice(2, 1)
    with pytest.raises(KeyError):
        feed.view(0).bar("Z")


def test_csv_round_trip_long_and_directory(tmp_path: Path) -> None:
    feed = two_symbol_feed()
    frames = {s: feed.symbol_frame(s) for s in feed.symbols}
    path = save_csv(frames, tmp_path / "long.csv")
    again = DataFeed.from_csv(path)
    np.testing.assert_allclose(again.array("close"), feed.array("close"))
    # directory of per-symbol files without a symbol column (stem = symbol)
    directory = tmp_path / "dir"
    directory.mkdir()
    for s, df in frames.items():
        out = df.copy()
        out.index.name = "Date"
        out.columns = [c.upper() for c in out.columns]
        out.to_csv(directory / f"{s}.csv")
    loaded = load_csv(directory, date_column="Date")
    assert sorted(loaded) == ["A", "B"]
    np.testing.assert_allclose(DataFeed(loaded).array("open"), feed.array("open"))


def test_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        load_csv(tmp_path / "missing.csv")
    with pytest.raises(DataError):
        load_csv(tmp_path)  # empty directory
    bad = tmp_path / "bad.csv"
    bad.write_text("when,open,high,low,close,volume\n2024-01-01,1,1,1,1,1\n")
    with pytest.raises(DataError, match="date"):
        load_csv(bad)


def test_from_long_frame() -> None:
    long = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Symbol": ["A", "B", "A"],
            "Open": [1, 2, 1],
            "High": [1, 2, 1],
            "Low": [1, 2, 1],
            "Close": [1, 2, 1],
            "Volume": [1, 1, 1],
        }
    )
    feed = DataFeed.from_long(long)
    assert feed.symbols == ("A", "B")
    assert feed.has_bar.sum() == 3
