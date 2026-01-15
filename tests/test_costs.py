from __future__ import annotations

import math

import pandas as pd
import pytest

from backtester import (
    BpsCommission,
    ConfigError,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    PerShareCommission,
    Side,
    SquareRootImpactSlippage,
)
from backtester.data import Bar

BAR = Bar(pd.Timestamp("2024-01-02"), "A", 100.0, 104.0, 96.0, 101.0, 1_000_000.0)


def test_fixed_bps_slippage() -> None:
    model = FixedBpsSlippage(10)
    assert model.fill_price(100.0, Side.BUY, 1, BAR) == pytest.approx(100.10)
    assert model.fill_price(100.0, Side.SELL, 1, BAR) == pytest.approx(99.90)
    assert NoSlippage().fill_price(100.0, Side.BUY, 1, BAR) == 100.0


def test_square_root_impact_with_given_volatility() -> None:
    model = SquareRootImpactSlippage(eta=0.5, volatility=0.02, spread_bps=4)
    # 2 bps half spread + 0.5 * 2% * sqrt(10_000 / 1_000_000) = 0.0002 + 0.001
    assert model.impact(Side.BUY, 10_000, BAR) == pytest.approx(0.0012)
    assert model.fill_price(100.0, Side.BUY, 10_000, BAR) == pytest.approx(100.12)
    assert model.fill_price(100.0, Side.SELL, 10_000, BAR) == pytest.approx(99.88)


def test_square_root_impact_parkinson_estimate_and_cap() -> None:
    model = SquareRootImpactSlippage(eta=1.0)
    sigma = math.log(104 / 96) / math.sqrt(4 * math.log(2))
    assert model.daily_volatility(BAR) == pytest.approx(sigma)
    assert model.impact(Side.BUY, 40_000, BAR) == pytest.approx(sigma * 0.2)
    capped = SquareRootImpactSlippage(eta=100.0, max_impact=0.05)
    assert capped.impact(Side.BUY, 1_000_000, BAR) == 0.05
    empty = Bar(BAR.timestamp, "A", 100, 100, 100, 100, 0.0)
    assert capped.impact(Side.BUY, 1, empty) == 0.05


def test_per_share_commission() -> None:
    model = PerShareCommission(rate=0.005, minimum=1.0, max_fraction=0.01)
    assert model.commission(100, 50.0) == 1.0  # minimum applies
    assert model.commission(1_000, 50.0) == 5.0
    assert model.commission(1_000, 0.10) == pytest.approx(1.0)  # capped at 1% of 100 notional
    assert model.commission(0, 50.0) == 0.0
    uncapped = PerShareCommission(rate=0.01, minimum=0.0, max_fraction=None)
    assert uncapped.commission(10, 0.01) == pytest.approx(0.1)


def test_bps_commission() -> None:
    assert BpsCommission(5).commission(200, 50.0) == pytest.approx(5.0)
    assert NoCommission().commission(1e9, 1e9) == 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FixedBpsSlippage(-1),
        lambda: SquareRootImpactSlippage(eta=-1),
        lambda: SquareRootImpactSlippage(volatility=-0.1),
        lambda: PerShareCommission(rate=-1),
        lambda: PerShareCommission(max_fraction=0),
        lambda: BpsCommission(-2),
    ],
)
def test_invalid_cost_parameters(factory) -> None:
    with pytest.raises(ConfigError):
        factory()
