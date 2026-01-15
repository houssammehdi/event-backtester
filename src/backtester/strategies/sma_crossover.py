"""Dual moving-average crossover (trend following)."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from backtester.errors import ConfigError
from backtester.strategy.rebalancing import FloatArray, Rebalance, TargetWeightStrategy


class SmaCrossover(TargetWeightStrategy):
    """Long (optionally short) each asset while its fast SMA is above its slow SMA.

    Each asset gets ``1 / N`` of equity when its signal is on, so the portfolio is at
    most 100 % gross invested. Signals are computed on closes and traded at the next
    open.

    Args:
        fast: Fast moving-average window (bars).
        slow: Slow moving-average window (bars).
        allow_short: Short assets whose fast SMA is below the slow SMA.
        rebalance: Rebalancing schedule (default: whenever a signal flips).
    """

    name: ClassVar[str] = "sma"

    def __init__(
        self,
        fast: int = 50,
        slow: int = 200,
        *,
        allow_short: bool = False,
        rebalance: Rebalance = "change",
    ) -> None:
        if not 0 < fast < slow:
            raise ConfigError("need 0 < fast < slow")
        super().__init__(rebalance)
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    @property
    def history_bars(self) -> int:
        """The slow window."""
        return self.slow

    def _signal(self, fast_ma: FloatArray, slow_ma: FloatArray) -> FloatArray:
        with np.errstate(invalid="ignore"):
            up = (fast_ma > slow_ma).astype(np.float64)
            down = (fast_ma < slow_ma).astype(np.float64)
        return up - down if self.allow_short else up

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Equal-weight long (short) positions in assets with a bullish (bearish) cross."""
        if len(closes) < self.slow:
            return None
        signal = self._signal(closes[-self.fast :].mean(axis=0), closes.mean(axis=0))
        weights: FloatArray = signal / closes.shape[1]
        return weights

    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Rolling-mean version of :meth:`desired_weights`."""
        fast = closes.rolling(self.fast).mean().to_numpy()
        slow = closes.rolling(self.slow).mean().to_numpy()
        weights = self._signal(fast, slow) / closes.shape[1]
        weights[: self.slow - 1] = np.nan
        return pd.DataFrame(weights, index=closes.index, columns=closes.columns)
