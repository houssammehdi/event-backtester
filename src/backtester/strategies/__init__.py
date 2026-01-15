"""Ready-made example strategies and a name registry used by the CLI."""

from __future__ import annotations

from backtester.strategies.bollinger import BollingerMeanReversion
from backtester.strategies.sma_crossover import SmaCrossover
from backtester.strategies.tsmom import TimeSeriesMomentum
from backtester.strategies.xsmom import CrossSectionalMomentum
from backtester.strategy.rebalancing import TargetWeightStrategy

STRATEGIES: dict[str, type[TargetWeightStrategy]] = {
    SmaCrossover.name: SmaCrossover,
    TimeSeriesMomentum.name: TimeSeriesMomentum,
    CrossSectionalMomentum.name: CrossSectionalMomentum,
    BollingerMeanReversion.name: BollingerMeanReversion,
}

__all__ = [
    "STRATEGIES",
    "BollingerMeanReversion",
    "CrossSectionalMomentum",
    "SmaCrossover",
    "TimeSeriesMomentum",
]
