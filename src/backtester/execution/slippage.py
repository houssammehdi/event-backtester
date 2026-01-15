"""Slippage models: how far the executed price deviates from the reference price."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from backtester.data.feed import Bar
from backtester.errors import ConfigError
from backtester.orders import Side


class SlippageModel(ABC):
    """Maps a reference price to an executed price, always adversely."""

    @abstractmethod
    def impact(self, side: Side, quantity: float, bar: Bar) -> float:
        """Adverse price move as a *fraction* of the reference price (``>= 0``)."""

    def fill_price(self, reference: float, side: Side, quantity: float, bar: Bar) -> float:
        """Executed price: buys pay ``ref * (1 + impact)``, sells get ``ref * (1 - impact)``."""
        frac = self.impact(side, quantity, bar)
        return reference * (1.0 + side.sign * frac)


@dataclass(frozen=True, slots=True)
class NoSlippage(SlippageModel):
    """Execute exactly at the reference price."""

    def impact(self, side: Side, quantity: float, bar: Bar) -> float:
        """Always zero."""
        return 0.0


@dataclass(frozen=True, slots=True)
class FixedBpsSlippage(SlippageModel):
    """Constant adverse slippage of ``bps`` basis points (e.g. half the bid-ask spread)."""

    bps: float = 5.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ConfigError("slippage bps must be non-negative")

    def impact(self, side: Side, quantity: float, bar: Bar) -> float:
        """``bps / 10_000`` regardless of size."""
        return self.bps / 10_000.0


@dataclass(frozen=True, slots=True)
class SquareRootImpactSlippage(SlippageModel):
    """Half-spread plus square-root market impact.

    ``impact = spread_bps / 2 / 1e4 + eta * sigma * sqrt(quantity / volume)``

    where ``sigma`` is the daily volatility. If ``volatility`` is ``None`` it is
    estimated from the executing bar's range with the Parkinson estimator
    ``ln(high / low) / sqrt(4 ln 2)``. The square-root law is the standard empirical
    model of the price impact of a metaorder (Almgren et al., 2005; Toth et al., 2011).
    """

    eta: float = 1.0
    volatility: float | None = None
    spread_bps: float = 0.0
    max_impact: float = 0.1

    def __post_init__(self) -> None:
        if self.eta < 0 or self.spread_bps < 0 or self.max_impact <= 0:
            raise ConfigError("eta and spread_bps must be >= 0 and max_impact > 0")
        if self.volatility is not None and self.volatility < 0:
            raise ConfigError("volatility must be non-negative")

    def daily_volatility(self, bar: Bar) -> float:
        """Volatility used for the impact term."""
        if self.volatility is not None:
            return self.volatility
        return math.log(bar.high / bar.low) / math.sqrt(4.0 * math.log(2.0))

    def impact(self, side: Side, quantity: float, bar: Bar) -> float:
        """Half-spread plus ``eta * sigma * sqrt(participation)``, capped at ``max_impact``."""
        half_spread = self.spread_bps / 20_000.0
        if bar.volume <= 0:
            return self.max_impact
        participation = quantity / bar.volume
        value = half_spread + self.eta * self.daily_volatility(bar) * math.sqrt(participation)
        return min(value, self.max_impact)
