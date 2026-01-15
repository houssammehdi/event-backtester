from __future__ import annotations

import math

import pandas as pd
import pytest

from backtester.analytics import round_trips

IDX = pd.bdate_range("2024-01-01", periods=10)


def fills(*rows: tuple[int, str, str, float, float, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": IDX[t],
                "order_id": k,
                "symbol": s,
                "side": side,
                "quantity": q,
                "price": p,
                "commission": c,
                "slippage": 0.0,
            }
            for k, (t, s, side, q, p, c) in enumerate(rows)
        ]
    )


PRICES = pd.DataFrame({"A": [100.0] * 10, "B": [50.0] * 10}, index=IDX)


def test_long_round_trip_with_scaling() -> None:
    log = fills(
        (0, "A", "buy", 10, 100, 1),
        (1, "A", "buy", 10, 110, 1),
        (2, "A", "sell", 5, 120, 1),
        (4, "A", "sell", 15, 90, 1),
    )
    t = round_trips(log, PRICES)
    assert len(t) == 1
    row = t.iloc[0]
    assert row["direction"] == "long"
    assert row["status"] == "closed"
    assert row["quantity"] == 20
    assert row["entry_price"] == pytest.approx(105)
    assert row["exit_price"] == pytest.approx((5 * 120 + 15 * 90) / 20)
    assert row["pnl"] == pytest.approx(5 * 15 + 15 * -15)
    assert row["net_pnl"] == pytest.approx(-150 - 4)
    assert row["bars_held"] == 4
    assert row["return"] == pytest.approx(-154 / 2100)


def test_flip_splits_into_two_trips_with_prorated_commission() -> None:
    log = fills(
        (0, "A", "buy", 10, 100, 0), (2, "A", "sell", 30, 110, 3), (5, "A", "buy", 20, 100, 2)
    )
    t = round_trips(log, PRICES)
    assert list(t["direction"]) == ["long", "short"]
    long, short = t.iloc[0], t.iloc[1]
    assert long["pnl"] == pytest.approx(100)
    assert long["commission"] == pytest.approx(1)  # 10 of the 30 shares
    assert short["quantity"] == 20
    assert short["pnl"] == pytest.approx(200)
    assert short["commission"] == pytest.approx(2 + 2)
    assert short["entry_time"] == IDX[2]


def test_open_trade_is_marked_at_last_price() -> None:
    log = fills((0, "B", "sell", 10, 60, 0))
    t = round_trips(log, PRICES)
    row = t.iloc[0]
    assert row["status"] == "open"
    assert pd.isna(row["exit_time"])
    assert math.isnan(row["exit_price"])
    assert row["pnl"] == pytest.approx(100)  # short from 60, marked at 50
    assert row["bars_held"] == 9


def test_empty_fill_log() -> None:
    assert round_trips(fills(), PRICES).empty
