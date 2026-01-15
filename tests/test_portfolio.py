"""Portfolio accounting, including property-based checks of the accounting identities."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester import ConfigError, FillEvent, Portfolio, Side

T = pd.Timestamp("2024-01-02")


def fill(symbol: str, qty: float, price: float, commission: float = 0.0) -> FillEvent:
    side = Side.BUY if qty > 0 else Side.SELL
    return FillEvent(T, 0, symbol, side, abs(qty), price, commission)


def test_average_cost_realized_pnl_and_flip() -> None:
    p = Portfolio(10_000, ["A"])
    p.on_fill(fill("A", 100, 10))
    p.on_fill(fill("A", 100, 12))
    pos = p.position("A")
    assert pos.avg_cost == 11
    assert p.on_fill(fill("A", -150, 13)) == pytest.approx(300)  # (13 - 11) * 150
    assert pos.quantity == 50
    assert pos.avg_cost == 11  # reducing does not move the average cost
    assert p.on_fill(fill("A", -100, 9)) == pytest.approx(-100)  # closes 50 at -2
    assert pos.quantity == -50
    assert pos.avg_cost == 9  # the flipped remainder is opened at the fill price
    p.mark({"A": 8})
    assert pos.unrealized_pnl == pytest.approx(50)
    assert p.realized_pnl == pytest.approx(200)
    assert p.cash == pytest.approx(10_000 - 1000 - 1200 + 1950 + 900)
    assert p.equity == pytest.approx(p.cash - 50 * 8)
    assert p.identity_error() < 1e-9


def test_short_round_trip_and_commissions() -> None:
    p = Portfolio(1_000)
    p.on_fill(fill("A", -10, 100, commission=1.0))
    assert p.cash == pytest.approx(1_999)
    p.mark({"A": 110})
    assert p.equity == pytest.approx(1_999 - 1_100)
    p.on_fill(fill("A", 10, 90, commission=1.0))
    assert p.position("A").is_flat
    assert p.realized_pnl == pytest.approx(100)
    assert p.equity == pytest.approx(1_000 + 100 - 2)
    assert p.total_commission == 2
    assert p.traded_notional == pytest.approx(1_900)


def test_exposures_weights_and_leverage() -> None:
    p = Portfolio(1_000, ["A", "B"])
    p.on_fill(fill("A", 10, 100))  # long 1000
    p.on_fill(fill("B", -5, 100))  # short 500
    p.mark({"A": 100, "B": 100, "C": math.nan})
    assert p.equity == pytest.approx(1_000)
    assert p.gross_exposure == pytest.approx(1_500)
    assert p.net_exposure == pytest.approx(500)
    assert p.leverage == pytest.approx(1.5)
    assert p.weights() == pytest.approx({"A": 1.0, "B": -0.5})


def test_borrow_fee_is_charged_on_shorts_only() -> None:
    p = Portfolio(10_000)
    p.on_fill(fill("A", 10, 100))
    p.on_fill(fill("B", -20, 50))
    fee = p.charge_borrow(0.0252, 1 / 252)
    assert fee == pytest.approx(1_000 * 0.0001)
    assert p.total_borrow_cost == pytest.approx(fee)
    assert p.identity_error() < 1e-9
    assert p.charge_borrow(0.0, 1.0) == 0.0


def test_invalid_initial_cash() -> None:
    with pytest.raises(ConfigError):
        Portfolio(0)


fills_strategy = st.lists(
    st.tuples(
        st.sampled_from(["A", "B", "C"]),
        st.integers(min_value=-500, max_value=500).filter(bool),
        st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=300, deadline=None)
@given(fills_strategy)
def test_accounting_identities_hold_for_any_fill_sequence(steps) -> None:
    p = Portfolio(100_000)
    cash_flows = 0.0
    marks: dict[str, float] = {}
    quantities: dict[str, float] = {}
    for symbol, qty, price, commission, drift in steps:
        p.on_fill(fill(symbol, qty, price, commission))
        cash_flows -= qty * price + commission
        quantities[symbol] = quantities.get(symbol, 0.0) + qty
        marks = {**marks, symbol: price * drift}
        p.mark(marks)
        # identity 1: equity = cash + sum(qty * price), with independently tracked inputs
        direct = 100_000 + cash_flows + sum(q * marks[s] for s, q in quantities.items())
        assert p.equity == pytest.approx(direct, rel=1e-12, abs=1e-6)
        assert p.cash == pytest.approx(100_000 + cash_flows, rel=1e-12, abs=1e-6)
        # identity 2: equity = initial + realized + unrealized - costs
        assert p.identity_error() < 1e-6
        for s, q in quantities.items():
            assert p.quantity(s) == pytest.approx(q)


@settings(max_examples=200, deadline=None)
@given(fills_strategy)
def test_flattening_realizes_everything(steps) -> None:
    p = Portfolio(100_000)
    for symbol, qty, price, commission, _ in steps:
        p.on_fill(fill(symbol, qty, price, commission))
    for symbol, pos in list(p.positions.items()):
        if not pos.is_flat:
            p.on_fill(fill(symbol, -pos.quantity, 100.0))
    assert p.unrealized_pnl == 0
    assert p.market_value == 0
    assert p.equity == pytest.approx(100_000 + p.realized_pnl - p.total_commission, abs=1e-6)
