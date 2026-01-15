"""Time-series momentum with volatility targeting."""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pandas as pd

from backtester.errors import ConfigError
from backtester.strategy.rebalancing import FloatArray, Rebalance, TargetWeightStrategy


class TimeSeriesMomentum(TargetWeightStrategy):
    """Go long assets with positive trailing returns, short those with negative ones.

    Positions are volatility-targeted in the spirit of Moskowitz, Ooi & Pedersen
    (2012): each asset's weight is ``sign(r_lookback) * (target_vol / sigma) / N`` with
    ``sigma`` the annualised volatility of daily log returns over ``vol_lookback``
    bars. Weights are capped per asset and the book is scaled down to ``max_gross``.

    Args:
        lookback: Momentum look-back in bars (252 = 12 months).
        vol_lookback: Window for the volatility estimate.
        target_vol: Annualised volatility target per asset before the ``1 / N`` split.
        max_weight: Cap on any single absolute weight.
        max_gross: Cap on gross leverage of the whole book.
        allow_short: If false, negative-momentum assets are held flat.
        rebalance: Rebalancing schedule (default monthly).
    """

    name: ClassVar[str] = "tsmom"

    def __init__(
        self,
        lookback: int = 252,
        vol_lookback: int = 63,
        target_vol: float = 0.15,
        *,
        max_weight: float = 1.0,
        max_gross: float = 2.0,
        allow_short: bool = True,
        rebalance: Rebalance = "monthly",
    ) -> None:
        if lookback < 2 or vol_lookback < 2:
            raise ConfigError("lookback and vol_lookback must be >= 2")
        if target_vol <= 0 or max_weight <= 0 or max_gross <= 0:
            raise ConfigError("target_vol, max_weight and max_gross must be positive")
        super().__init__(rebalance)
        self.lookback = lookback
        self.vol_lookback = vol_lookback
        self.target_vol = target_vol
        self.max_weight = max_weight
        self.max_gross = max_gross
        self.allow_short = allow_short

    @property
    def history_bars(self) -> int:
        """Enough bars for both the momentum return and the volatility window."""
        return max(self.lookback, self.vol_lookback) + 1

    def _combine(self, momentum: FloatArray, vol: FloatArray, n: int) -> FloatArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            signal = np.sign(momentum)
            if not self.allow_short:
                signal = np.maximum(signal, 0.0)
            raw = signal * np.minimum(self.target_vol / vol, n * self.max_weight) / n
        raw = np.where(np.isfinite(raw), raw, 0.0)
        gross = np.abs(raw).sum(axis=-1, keepdims=True)
        scale = np.where(gross > self.max_gross, self.max_gross / np.maximum(gross, 1e-300), 1.0)
        out: FloatArray = raw * scale
        return out

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Volatility-scaled sign of the trailing return."""
        if len(closes) < self.history_bars:
            return None
        momentum = closes[-1] / closes[-1 - self.lookback] - 1.0
        log_ret = np.diff(np.log(closes[-self.vol_lookback - 1 :]), axis=0)
        vol = log_ret.std(axis=0, ddof=1) * math.sqrt(252.0)
        return self._combine(momentum, vol, closes.shape[1])

    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Bulk version using shifted prices and a rolling standard deviation."""
        momentum = (closes / closes.shift(self.lookback) - 1.0).to_numpy()
        log_ret = pd.DataFrame(np.log(closes.to_numpy())).diff()
        vol = (log_ret.rolling(self.vol_lookback).std() * math.sqrt(252.0)).to_numpy()
        weights = self._combine(momentum, vol, closes.shape[1])
        weights[: self.history_bars - 1] = np.nan
        return pd.DataFrame(weights, index=closes.index, columns=closes.columns)
