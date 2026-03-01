"""Base class for strategies that express decisions as target portfolio weights."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.data.feed import DataFeed, MarketView
from backtester.errors import ConfigError
from backtester.strategy.base import StrategyContext, VectorizedStrategy

Rebalance = Literal["daily", "monthly", "change"]
FloatArray = npt.NDArray[np.float64]


class TargetWeightStrategy(VectorizedStrategy):
    """Strategy defined by a per-bar weight rule plus a rebalancing schedule.

    Subclasses implement the rule twice - incrementally in :meth:`desired_weights`
    (used by the event engine, fed only the read-only view's history) and in bulk in
    :meth:`desired_weights_frame` (used by the vectorized fast path). The parity tests
    check the two agree, which doubles as a look-ahead check of the bulk version.

    Rebalancing schedules:

    * ``"daily"`` - re-target on every bar (trades away drift);
    * ``"monthly"`` - on the first bar of each calendar month, detected from the current
      and previous timestamps only (no peeking at the next bar);
    * ``"change"`` - whenever the desired weights differ from the previous bar's.

    The first bar with valid weights is always a rebalance. When a backtest starts
    after a warm-up period, the most recently scheduled weights are established on the
    first trading bar.

    Subclasses whose weights depend only on the trailing window - no state carried from
    bar to bar - may set :attr:`stateless`. With the monthly schedule both paths then
    evaluate :meth:`desired_weights` only on bars where a rebalance can be due (every
    bar until the first valid weights, then the first bar of each month). The decisions
    are the same; on daily data the rule is evaluated about 20 times less often, which
    matters for expensive rules such as portfolio optimisers.
    """

    stateless: ClassVar[bool] = False
    """True if :meth:`desired_weights` is a pure function of its window."""

    def __init__(self, rebalance: Rebalance) -> None:
        if rebalance not in ("daily", "monthly", "change"):
            raise ConfigError(f"unknown rebalance schedule {rebalance!r}")
        self.rebalance: Rebalance = rebalance
        self._previous: FloatArray | None = None
        self._scheduled: FloatArray | None = None
        self._emitted = False

    @property
    @abstractmethod
    def history_bars(self) -> int:
        """Number of trailing bars :meth:`desired_weights` needs."""

    @abstractmethod
    def desired_weights(self, closes: FloatArray) -> FloatArray | None:
        """Weights for the latest row of ``closes`` (forward-filled, oldest first).

        Return ``None`` while there is not enough history. Called on every bar in order
        (unless :attr:`stateless`), so implementations may keep state.
        """

    @abstractmethod
    def desired_weights_frame(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Bulk version of :meth:`desired_weights` for every row (NaN rows = no signal)."""

    # ------------------------------------------------------------------ event path
    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        """Compute weights, decide whether a rebalance is due and emit the target."""
        due = False
        if not self._skip(view):
            weights = self.desired_weights(view.filled_closes(self.history_bars))
            if weights is not None:
                due = self._is_due(view, weights)
                if due:
                    self._scheduled = weights
                self._previous = weights
        if ctx.is_warmup or self._scheduled is None:
            return
        if due or not self._emitted:
            ctx.target_weights(
                {s: float(w) for s, w in zip(view.symbols, self._scheduled, strict=True)}
            )
            self._emitted = True

    def _skip(self, view: MarketView) -> bool:
        """Whether a stateless monthly strategy can skip this bar entirely."""
        return (
            self.stateless
            and self.rebalance == "monthly"
            and self._previous is not None
            and not self._month_changed(view)
        )

    @staticmethod
    def _month_changed(view: MarketView) -> bool:
        before = view.previous_timestamp
        if before is None:
            return True
        now = view.timestamp
        return (now.year, now.month) != (before.year, before.month)

    def _is_due(self, view: MarketView, weights: FloatArray) -> bool:
        if self._previous is None or self.rebalance == "daily":
            return True
        if self.rebalance == "monthly":
            return self._month_changed(view)
        return not np.array_equal(weights, self._previous)

    # ------------------------------------------------------------------ vectorized path
    def _desired_on_candidates(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Desired weights on the rows where a monthly rebalance can be due, else NaN."""
        values = closes.to_numpy(dtype=np.float64)
        periods = pd.DatetimeIndex(closes.index).to_period("M")
        month_change = np.r_[True, periods[1:] != periods[:-1]]
        out = np.full(values.shape, np.nan)
        found = False
        for t in range(len(values)):
            if found and not month_change[t]:
                continue
            weights = self.desired_weights(values[max(0, t + 1 - self.history_bars) : t + 1])
            if weights is not None:
                out[t] = weights
                found = True
        return pd.DataFrame(out, index=closes.index, columns=closes.columns)

    def target_weights_frame(self, feed: DataFeed) -> pd.DataFrame:
        """Scheduled target weights per bar; NaN rows mean *hold, do not rebalance*."""
        closes = pd.DataFrame(feed.last_close.copy(), index=feed.index, columns=feed.symbols)
        if self.stateless and self.rebalance == "monthly":
            desired = self._desired_on_candidates(closes)
        else:
            desired = self.desired_weights_frame(closes)
        valid = desired.notna().all(axis=1).to_numpy()
        first_valid = np.zeros(len(desired), dtype=bool)
        if valid.any():
            first_valid[int(np.argmax(valid))] = True
        if self.rebalance == "daily":
            due = valid.copy()
        elif self.rebalance == "monthly":
            periods = pd.DatetimeIndex(desired.index).to_period("M")
            month_change = np.r_[False, periods[1:] != periods[:-1]]
            due = valid & (month_change | first_valid)
        else:
            values = desired.to_numpy()
            changed = np.r_[False, ~np.all(values[1:] == values[:-1], axis=1)]
            due = valid & (changed | first_valid)
        return desired.where(pd.Series(due, index=desired.index), axis=0)
