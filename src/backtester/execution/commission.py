"""Commission models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backtester.errors import ConfigError


class CommissionModel(ABC):
    """Computes the fee charged for one fill."""

    @abstractmethod
    def commission(self, quantity: float, price: float) -> float:
        """Fee in currency units for trading ``quantity`` (absolute) at ``price``."""


@dataclass(frozen=True, slots=True)
class NoCommission(CommissionModel):
    """Free trading."""

    def commission(self, quantity: float, price: float) -> float:
        """Always zero."""
        return 0.0


@dataclass(frozen=True, slots=True)
class PerShareCommission(CommissionModel):
    """Per-share fee with a per-order minimum and an optional cap as a share of notional.

    ``fee = min(max(rate * quantity, minimum), max_fraction * quantity * price)``
    (the cap applies only when ``max_fraction`` is set), mirroring common retail
    brokerage schedules.
    """

    rate: float = 0.005
    minimum: float = 1.0
    max_fraction: float | None = 0.01

    def __post_init__(self) -> None:
        if self.rate < 0 or self.minimum < 0:
            raise ConfigError("rate and minimum must be non-negative")
        if self.max_fraction is not None and self.max_fraction <= 0:
            raise ConfigError("max_fraction must be positive")

    def commission(self, quantity: float, price: float) -> float:
        """Per-share fee with minimum and optional notional cap."""
        if quantity <= 0:
            return 0.0
        fee = max(self.rate * quantity, self.minimum)
        if self.max_fraction is not None:
            fee = min(fee, self.max_fraction * quantity * price)
        return fee


@dataclass(frozen=True, slots=True)
class BpsCommission(CommissionModel):
    """Fee proportional to traded notional: ``bps / 1e4 * quantity * price``."""

    bps: float = 1.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ConfigError("commission bps must be non-negative")

    def commission(self, quantity: float, price: float) -> float:
        """Proportional fee."""
        return self.bps / 10_000.0 * abs(quantity) * price
