"""Cross-sectional momentum rotation."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from backtester.errors import ConfigError
from backtester.strategy.rebalancing import FloatArray, Rebalance, TargetWeightStrategy


class CrossSectionalMomentum(TargetWeightStrategy):
    """Hold the ``top_k`` assets with the best trailing return, equally weighted.

    The ranking return runs from ``lookback`` bars ago to ``skip`` bars ago, skipping
    the most recent month to sidestep short-term reversal (Jegadeesh & Titman, 1993).
    Ties are broken by symbol order, so results are deterministic.

    Args:
        lookback: Start of the ranking window, in bars before today.
        skip: End of the ranking window, in bars before today (``0`` = today).
        top_k: Number of assets held long.
        long_short: Also short the ``top_k`` worst assets (each ``-1 / top_k``).
        rebalance: Rebalancing schedule (default monthly).
    """

    name: ClassVar[str] = "xsmom"

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        top_k: int = 2,
        *,
        long_short: bool = False,
        rebalance: Rebalance = "monthly",
    ) -> None:
        if not 0 <= skip < lookback:
            raise ConfigError("need 0 <= skip < lookback")
        if top_k < 1:
            raise ConfigError("top_k must be >= 1")
        super().__init__(rebalance)
        self.lookback = lookback
        self.skip = skip
        self.top_k = top_k
        self.long_short = long_short

    @property
    def history_bars(self) -> int:
        """``lookback + 1`` bars."""
        return self.lookback + 1

    def _rank_weights(self, scores: FloatArray) -> FloatArray:
        """Rows of scores -> rows of weights (works for 1-D and 2-D input)."""
        s = np.atleast_2d(scores)
        weights = np.zeros_like(s)
        n_valid = np.sum(~np.isnan(s), axis=1)
        k = np.minimum(self.top_k, n_valid if not self.long_short else n_valid // 2)
        # Stable argsort on -score keeps symbol order among ties; NaN sorts last.
        order_desc = np.argsort(np.where(np.isnan(s), np.inf, -s), axis=1, kind="stable")
        order_asc = np.argsort(np.where(np.isnan(s), np.inf, s), axis=1, kind="stable")
        rows = np.arange(s.shape[0])
        for rank in range(self.top_k):
            active = rank < k
            weights[rows[active], order_desc[active, rank]] = 1.0 / self.top_k
            if self.long_short:
                weights[rows[active], order_asc[active, rank]] = -1.0 / self.top_k
        out: FloatArray = weights.reshape(np.shape(scores))
        return out

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Equal weights on the best (and optionally worst) ranked assets."""
        if len(closes) < self.history_bars:
            return None
        scores = closes[-1 - self.skip] / closes[-1 - self.lookback] - 1.0
        return self._rank_weights(scores)

    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Bulk version ranking every row at once."""
        scores = (closes.shift(self.skip) / closes.shift(self.lookback) - 1.0).to_numpy()
        weights = self._rank_weights(scores)
        weights[: self.history_bars - 1] = np.nan
        return pd.DataFrame(weights, index=closes.index, columns=closes.columns)
