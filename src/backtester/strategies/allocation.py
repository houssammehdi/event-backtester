"""Periodic rebalancing to a portfolio-construction method estimated on a trailing window."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from backtester.errors import ConfigError
from backtester.portfolio.allocation import ALLOCATORS, Allocator
from backtester.portfolio.covariance import LedoitWolf, SampleCovariance
from backtester.strategy.rebalancing import FloatArray, Rebalance, TargetWeightStrategy

CovarianceName = Literal["ledoit_wolf", "ledoit_wolf_identity", "sample"]
_COVARIANCES = {
    "ledoit_wolf": LedoitWolf("constant_correlation"),
    "ledoit_wolf_identity": LedoitWolf("identity"),
    "sample": SampleCovariance(),
}


class AllocationStrategy(TargetWeightStrategy):
    """Long-only allocation by EW, IV, minimum variance, mean-variance, ERC or HRP.

    On each rebalance date the allocator is applied to the simple returns of the last
    ``lookback`` bars (closes forward-filled over missing bars). Symbols without a
    complete window - not yet listed - get zero weight until they have one. The
    weights depend only on the window, so the strategy is :attr:`stateless` and, on
    the monthly schedule, the allocator runs only on the first bar of each month.

    Args:
        method: Allocator name (``"ew"``, ``"iv"``, ``"minvar"``, ``"meanvar"``,
            ``"erc"``, ``"hrp"``) or an :class:`~backtester.portfolio.Allocator`.
        lookback: Estimation window in bars.
        covariance: Covariance estimator for a named method: ``"ledoit_wolf"``
            (constant-correlation shrinkage, default), ``"ledoit_wolf_identity"`` or
            ``"sample"``.
        max_weight: Cap per asset for ``"minvar"`` and ``"meanvar"``.
        risk_aversion: Risk aversion of ``"meanvar"``.
        rebalance: Rebalancing schedule (default monthly).
    """

    name: ClassVar[str] = "allocation"
    stateless: ClassVar[bool] = True

    def __init__(
        self,
        method: str | Allocator = "hrp",
        lookback: int = 252,
        *,
        covariance: CovarianceName = "ledoit_wolf",
        max_weight: float | None = None,
        risk_aversion: float = 10.0,
        rebalance: Rebalance = "monthly",
    ) -> None:
        if lookback < 2:
            raise ConfigError("lookback must be at least 2 bars")
        if isinstance(method, str):
            if method not in ALLOCATORS:
                raise ConfigError(f"unknown allocation method {method!r}: {sorted(ALLOCATORS)}")
            if covariance not in _COVARIANCES:
                raise ConfigError(f"unknown covariance estimator {covariance!r}")
            cls = ALLOCATORS[method]
            options: dict[str, Any] = {"covariance": _COVARIANCES[covariance]}
            names = {f.name for f in fields(cls)}
            if "max_weight" in names:
                options["max_weight"] = max_weight
            if "risk_aversion" in names:
                options["risk_aversion"] = risk_aversion
            self._allocator: Allocator = cls(**options)
        else:
            self._allocator = method
        super().__init__(rebalance)
        self.method = method if isinstance(method, str) else type(method).name
        self.lookback = lookback
        self.covariance = covariance
        self.max_weight = max_weight
        self.risk_aversion = risk_aversion

    @property
    def history_bars(self) -> int:
        """``lookback`` returns need ``lookback + 1`` closes."""
        return self.lookback + 1

    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Allocator weights over the symbols with a complete window."""
        if len(closes) < self.history_bars:
            return None
        with np.errstate(invalid="ignore"):
            returns = closes[1:] / closes[:-1] - 1.0
        complete = ~np.isnan(returns).any(axis=0)
        weights = np.zeros(closes.shape[1])
        if complete.any():
            weights[complete] = self._allocator(returns[:, complete])
        return weights

    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Every row from :meth:`desired_weights` (the allocators are not vectorisable).

        The vectorized fast path does not call this for the monthly schedule: it
        evaluates only the rows where a rebalance can be due.
        """
        values = closes.to_numpy(dtype=np.float64)
        out = np.full(values.shape, np.nan)
        for t in range(len(values)):
            weights = self.desired_weights(values[max(0, t + 1 - self.history_bars) : t + 1])
            if weights is not None:
                out[t] = weights
        return pd.DataFrame(out, index=closes.index, columns=closes.columns)
