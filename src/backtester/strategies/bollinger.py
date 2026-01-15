"""Bollinger-band mean reversion."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from backtester.errors import ConfigError
from backtester.strategy.rebalancing import FloatArray, Rebalance, TargetWeightStrategy


class BollingerMeanReversion(TargetWeightStrategy):
    """Fade moves outside Bollinger bands and exit at the moving average.

    Per asset: go long when the close falls below ``SMA - n_std * sd``, stay long until
    the close is back at or above the SMA; symmetrically go short above the upper band
    (if ``allow_short``) until the close is back at or below the SMA. Each active asset
    gets ``1 / N`` of equity.

    Args:
        window: Band look-back (bars).
        n_std: Band width in sample standard deviations.
        allow_short: Trade the upper band as well.
        rebalance: Rebalancing schedule (default: whenever a position changes).
    """

    name: ClassVar[str] = "bollinger"

    def __init__(
        self,
        window: int = 20,
        n_std: float = 2.0,
        *,
        allow_short: bool = True,
        rebalance: Rebalance = "change",
    ) -> None:
        if window < 2 or n_std <= 0:
            raise ConfigError("need window >= 2 and n_std > 0")
        super().__init__(rebalance)
        self.window = window
        self.n_std = n_std
        self.allow_short = allow_short
        self._long: FloatArray | None = None
        self._short: FloatArray | None = None

    @property
    def history_bars(self) -> int:
        """The band window."""
        return self.window

    def _step(
        self,
        close: FloatArray,
        mean: FloatArray,
        sd: FloatArray,
        long: FloatArray,
        short: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        lower, upper = mean - self.n_std * sd, mean + self.n_std * sd
        with np.errstate(invalid="ignore"):
            long = np.where(close < lower, 1.0, np.where(close >= mean, 0.0, long))
            short = np.where(close > upper, -1.0, np.where(close <= mean, 0.0, short))
        if not self.allow_short:
            short = np.zeros_like(short)
        return long, short

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Advance the per-asset long/short state machine by one bar."""
        n = closes.shape[1]
        if self._long is None or self._short is None:
            self._long, self._short = np.zeros(n), np.zeros(n)
        if len(closes) < self.window:
            return None
        mean = closes.mean(axis=0)
        sd = closes.std(axis=0, ddof=1)
        self._long, self._short = self._step(closes[-1], mean, sd, self._long, self._short)
        weights: FloatArray = (self._long + self._short) / n
        return weights

    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Bulk version: the state machine is resolved by forward-filling entry/exit marks."""
        c = closes.to_numpy()
        mean = closes.rolling(self.window).mean().to_numpy()
        sd = closes.rolling(self.window).std(ddof=1).to_numpy()
        lower, upper = mean - self.n_std * sd, mean + self.n_std * sd
        with np.errstate(invalid="ignore"):
            long_marks = np.where(c < lower, 1.0, np.where(c >= mean, 0.0, np.nan))
            short_marks = np.where(c > upper, -1.0, np.where(c <= mean, 0.0, np.nan))
        long = pd.DataFrame(long_marks).ffill().fillna(0.0).to_numpy()
        short = pd.DataFrame(short_marks).ffill().fillna(0.0).to_numpy()
        if not self.allow_short:
            short = np.zeros_like(short)
        weights = (long + short) / closes.shape[1]
        weights[: self.window - 1] = np.nan
        return pd.DataFrame(weights, index=closes.index, columns=closes.columns)
