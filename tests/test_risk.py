from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester import (
    ConfigError,
    DataFeed,
    Engine,
    FillEvent,
    OrderEvent,
    Portfolio,
    RiskLimits,
    RiskManager,
    Side,
    SignalEvent,
    TargetEvent,
)
from tests.conftest import Scripted, bars

T = pd.Timestamp("2024-01-02")


def portfolio_with(qty: float = 0.0, price: float = 100.0) -> Portfolio:
    p = Portfolio(100_000, ["A", "B"])
    if qty:
        side = Side.BUY if qty > 0 else Side.SELL
        p.on_fill(FillEvent(T, 0, "A", side, abs(qty), price, 0.0))
    p.mark({"A": price, "B": 50.0})
    return p


def signal(qty: float, order_id: int = 1) -> SignalEvent:
    return SignalEvent(T, order_id=order_id, symbol="A", quantity=qty)


def test_limits_validation() -> None:
    with pytest.raises(ConfigError):
        RiskLimits(max_position_weight=0)
    with pytest.raises(ConfigError):
        RiskLimits(max_drawdown=1.5)
    with pytest.raises(ConfigError):
        RiskManager(rebalance_threshold=-1)


def test_constrain_weights() -> None:
    rm = RiskManager(RiskLimits(max_position_weight=0.4, max_gross_leverage=1.0, allow_short=False))
    out = rm.constrain_weights({"A": 0.9, "B": 0.5, "C": -0.3, "D": float("nan")})
    # clip to 0.4 each, drop the short and the NaN, then scale 0.8 gross -> fine (<= 1)
    assert out == pytest.approx({"A": 0.4, "B": 0.4, "C": 0.0, "D": 0.0})
    rm2 = RiskManager(RiskLimits(max_gross_leverage=1.0))
    assert rm2.constrain_weights({"A": 1.0, "B": -1.0}) == pytest.approx({"A": 0.5, "B": -0.5})


def test_target_translation_cancels_and_sizes_orders() -> None:
    rm = RiskManager(RiskLimits(max_position_weight=0.3))
    p = portfolio_with(qty=100)
    ids = iter(range(10, 20))
    events = rm.process_target(
        TargetEvent(T, weights={"A": 0.5, "B": 0.2}), p, {"A": 100.0, "B": 50.0}, ids.__next__
    )
    orders = [e for e in events if isinstance(e, OrderEvent)]
    assert type(events[0]).__name__ == "CancelEvent"
    # A clipped to 30% of 100k = 300 shares (hold 100 -> buy 200); B 20% = 400 shares
    assert {(o.symbol, o.side, o.quantity) for o in orders} == {
        ("A", Side.BUY, 200),
        ("B", Side.BUY, 400),
    }
    assert any(a.kind == "clipped" for a in rm.log)


def test_signal_is_clipped_to_position_limit_counting_pending_orders() -> None:
    rm = RiskManager(RiskLimits(max_position_weight=0.2))  # 20k -> 200 shares at 100
    p = portfolio_with(qty=100)
    approved, ok = rm.review_signal(signal(500), p, {"A": 100.0}, pending={"A": 50.0})
    assert ok
    assert approved.quantity == 50  # 100 held + 50 pending + 50 = 200
    rejected, ok = rm.review_signal(signal(10), p, {"A": 100.0}, pending={"A": 100.0})
    assert not ok
    assert rejected.quantity == 10


def test_reducing_orders_are_always_allowed() -> None:
    rm = RiskManager(RiskLimits(max_position_weight=0.01, max_gross_leverage=0.01))
    p = portfolio_with(qty=500)  # already far above the limits
    approved, ok = rm.review_signal(signal(-200), p, {"A": 100.0}, pending={})
    assert ok
    assert approved.quantity == 200
    assert approved.side is Side.SELL


def test_gross_leverage_cap_on_signals() -> None:
    rm = RiskManager(RiskLimits(max_gross_leverage=1.0))
    p = portfolio_with(qty=600)  # 60% gross
    approved, ok = rm.review_signal(signal(1_000), p, {"A": 100.0}, pending={})
    assert ok
    assert approved.quantity == 400  # up to 100% gross


def test_no_short_rule() -> None:
    rm = RiskManager(RiskLimits(allow_short=False))
    p = portfolio_with(qty=100)
    approved, ok = rm.review_signal(signal(-300), p, {"A": 100.0}, pending={})
    assert ok
    assert approved.quantity == 100  # can sell what is held, not more
    _, ok = rm.review_signal(signal(-10), portfolio_with(), {"A": 100.0}, pending={})
    assert not ok


def crash_feed() -> DataFeed:
    prices = [100, 100, 100, 95, 88, 80, 75, 70, 72, 74, 76]
    rows = [(p, p * 1.01, p * 0.99, p, 1e6) for p in prices]
    return DataFeed({"A": bars(rows)})


def test_drawdown_kill_switch_flattens_and_halts() -> None:
    feed = crash_feed()

    def buy(view, ctx) -> None:
        ctx.target_weights({"A": 1.0})

    # keeps asking for the full position every bar after the crash too
    strategy = Scripted(dict.fromkeys(range(len(feed)), buy))
    risk = RiskManager(RiskLimits(max_drawdown=0.15))
    result = Engine(feed, strategy, initial_cash=100_000, risk=risk, check_invariants=True).run()
    assert result.halted_at == feed.index[5]  # 80 / 100 - 1 = -20% at the close of bar 5
    kill = [o for o in result.orders if o.tag == "kill_switch"]
    assert len(kill) == 1
    fills_after = result.fills[result.fills["timestamp"] > feed.index[6]]
    assert fills_after.empty  # flattened at bar 6 open, nothing afterwards
    assert result.positions["A"].iloc[-1] == 0
    assert np.isclose(result.equity.iloc[-1], result.equity.iloc[6])
    assert any(a.kind == "kill_switch" for a in result.risk_log)


def test_engine_applies_gross_leverage_limit_to_targets() -> None:
    feed = crash_feed()
    strategy = Scripted({0: lambda view, ctx: ctx.target_weights({"A": 3.0})})
    result = Engine(feed, strategy, risk=RiskLimits(max_gross_leverage=1.5)).run()
    assert result.weights["A"].iloc[1] == pytest.approx(1.5, rel=0.01)
