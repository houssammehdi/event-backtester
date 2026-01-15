from __future__ import annotations

import math

import numpy as np
import pytest

from backtester import ConfigError
from backtester.portfolio import (
    fixed_fractional_quantity,
    fixed_risk_quantity,
    realized_volatility,
    rebalance_quantities,
    round_to_lot,
    scale_to_gross,
    volatility_target_weights,
)


def test_round_to_lot_rounds_toward_zero() -> None:
    assert round_to_lot(12.9, 1) == 12
    assert round_to_lot(-12.9, 1) == -12
    assert round_to_lot(149, 100) == 100
    assert round_to_lot(0.4, 1) == 0
    assert round_to_lot(3.7, None) == 3.7
    with pytest.raises(ConfigError):
        round_to_lot(1, 0)


def test_fixed_fractional_and_fixed_risk() -> None:
    assert fixed_fractional_quantity(100_000, 33.0, 0.1) == 303
    assert fixed_fractional_quantity(100_000, 33.0, -0.1) == -303
    assert fixed_fractional_quantity(100_000, 33.0, 0.1, lot_size=None) == pytest.approx(
        10_000 / 33
    )
    # risking 1% of 100k with a stop 2.5 below the entry -> 400 shares long
    assert fixed_risk_quantity(100_000, 50.0, 47.5, 0.01) == 400
    assert fixed_risk_quantity(100_000, 50.0, 52.5, 0.01) == -400
    with pytest.raises(ConfigError):
        fixed_fractional_quantity(1, 0.0, 0.1)
    with pytest.raises(ConfigError):
        fixed_risk_quantity(1, 10, 10, 0.1)


def test_volatility_targeting() -> None:
    rng = np.random.default_rng(0)
    returns = np.column_stack([rng.normal(0, 0.01, 500), rng.normal(0, 0.02, 500)])
    vol = realized_volatility(returns)
    expected = returns.std(axis=0, ddof=1) * math.sqrt(252)
    np.testing.assert_allclose(vol, expected)
    weights = volatility_target_weights(returns, 0.10)
    np.testing.assert_allclose(weights, 0.10 / expected)
    assert weights[0] == pytest.approx(2 * weights[1], rel=0.1)
    capped = volatility_target_weights(returns, 0.10, max_weight=0.5)
    assert capped.max() <= 0.5
    short_history = volatility_target_weights(np.array([[0.01], [np.nan]]), 0.1)
    assert short_history.tolist() == [0.0]
    with pytest.raises(ConfigError):
        volatility_target_weights(returns, 0.0)


def test_scale_to_gross() -> None:
    assert scale_to_gross({"A": 1.5, "B": -1.5}, 2.0) == pytest.approx({"A": 1.0, "B": -1.0})
    assert scale_to_gross({"A": 0.5}, 2.0) == {"A": 0.5}


def test_rebalance_quantities() -> None:
    prices = {"A": 100.0, "B": 50.0, "C": math.nan}
    orders = rebalance_quantities(
        {"A": 0.5, "B": 0.25, "C": 0.25}, {"A": 100, "D": 10}, prices, 100_000
    )
    # A: 500 target - 100 held; B: 500; C: no valid price; D: not in prices -> skipped
    assert orders == {"A": 400, "B": 500}


def test_rebalance_band_and_minimum_notional() -> None:
    prices = {"A": 100.0, "B": 100.0}
    held = {"A": 490.0, "B": 100.0}
    # A is 49% vs 50% target: inside a 2% band; B closes regardless of the band
    orders = rebalance_quantities({"A": 0.5}, held, prices, 100_000, threshold=0.02)
    assert orders == {"B": -100}
    none = rebalance_quantities({"A": 0.5}, {"A": 499.0}, prices, 100_000, min_notional=500)
    assert none == {}
