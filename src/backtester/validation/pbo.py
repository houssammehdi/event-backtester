"""Probability of backtest overfitting via combinatorially symmetric cross-validation.

Given the per-period returns of ``N`` configurations over the same ``T`` periods (the
*performance matrix* of a parameter search), CSCV asks: when the configuration that
looks best on one half of the data is evaluated on the other half, how often does it
fall below the median configuration? (Bailey, Borwein, Lopez de Prado & Zhu, 2017.)

1. Split the rows into ``S`` (even) contiguous blocks of equal size.
2. For every one of the ``C(S, S/2)`` ways to pick half of the blocks as the training
   set ``J`` (the rest, ``J_bar``, is the test set), compute the performance of every
   configuration on ``J`` and on ``J_bar``.
3. Select ``n* = argmax`` of the training performance, and find its relative rank
   ``w = rank / (N + 1)`` among the test performances (1 = worst, ``N`` = best).
4. The logit ``lambda = ln(w / (1 - w))`` is negative when the selected configuration
   ends up below the median out of sample.

``PBO = P(lambda <= 0)`` over all combinations. A family of pure-noise strategies gives
``PBO = 1/2`` (for even ``N``); a search that finds genuine, persistent skill gives a
PBO near 0. The regression of test on training performance of the selected
configuration measures *performance degradation*, and the share of combinations in
which it loses money out of sample is the *probability of loss*.

Every combination uses the same rows exactly once in the training or the test set, so
the performance statistics come from block sums and are computed for all combinations
with two matrix products.

Reference:
    Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2017). The
    probability of backtest overfitting. *Journal of Computational Finance* 20(4),
    39-69.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Metric = Literal["sharpe", "mean", "sortino"]
MetricFunction = Callable[[FloatArray], FloatArray]
"""Maps a ``(rows, N)`` block of returns to one performance value per column."""


@dataclass(frozen=True, slots=True)
class PBOResult:
    """Output of :func:`probability_of_backtest_overfitting`.

    Arrays have one entry per combination of training blocks.
    """

    pbo: float
    """Share of combinations whose selected configuration ranks at or below the
    out-of-sample median (``logit <= 0``)."""
    logits: FloatArray = field(repr=False)
    relative_ranks: FloatArray = field(repr=False)
    """Out-of-sample relative rank ``w`` of the selected configuration, in (0, 1)."""
    is_performance: FloatArray = field(repr=False)
    """Training performance of the selected configuration."""
    oos_performance: FloatArray = field(repr=False)
    """Test performance of the selected configuration."""
    selected: IntArray = field(repr=False)
    """Column index of the selected configuration."""
    degradation_slope: float
    """OLS slope of test on training performance of the selected configuration."""
    degradation_intercept: float
    degradation_r2: float
    prob_oos_loss: float
    """Share of combinations in which the selected configuration's test
    performance is negative."""
    n_splits: int
    n_obs: int
    """Rows used (the oldest ``T mod S`` rows are dropped to equalise blocks)."""
    labels: tuple[str, ...]
    metric: str

    @property
    def n_combinations(self) -> int:
        """Number of train/test combinations, ``C(S, S/2)``."""
        return len(self.logits)

    @property
    def n_strategies(self) -> int:
        """Number of configurations compared."""
        return len(self.labels)

    def selection_frequency(self) -> pd.Series:
        """How often each configuration was the training-set winner (sums to 1)."""
        counts = np.bincount(self.selected, minlength=self.n_strategies)
        return pd.Series(counts / counts.sum(), index=list(self.labels), name="selected")


def _as_matrix(returns: npt.ArrayLike | pd.DataFrame) -> tuple[FloatArray, tuple[str, ...]]:
    if isinstance(returns, pd.DataFrame):
        labels = tuple(str(c) for c in returns.columns)
        values = returns.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("returns must be a 2-D (periods x configurations) array")
        labels = tuple(str(j) for j in range(values.shape[1]))
    if np.isnan(values).any():
        raise ValueError("returns contain NaN; align the configurations first")
    return values, labels


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """``numerator / denominator`` with 0/0 := 0 and x/0 := sign(x) * inf."""
    zero = denominator <= 0
    flat = np.where(numerator > 0, np.inf, np.where(numerator < 0, -np.inf, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        result: FloatArray = np.where(zero, flat, numerator / np.where(zero, 1.0, denominator))
    return result


def _block_metric(
    blocks: FloatArray, masks: FloatArray, metric: Metric
) -> tuple[FloatArray, FloatArray]:
    """Metric of every column on the training and test rows of every combination.

    ``blocks`` is ``(S, rows_per_block, N)``; ``masks`` is ``(C, S)`` with 1 marking a
    training block. Returns two ``(C, N)`` arrays.
    """
    n_rows = blocks.shape[1] * masks.shape[1] // 2
    centre = blocks.mean(axis=(0, 1))
    centred = blocks - centre
    s1 = centred.sum(axis=1)
    s2 = (centred**2).sum(axis=1)
    s_down = (np.minimum(blocks, 0.0) ** 2).sum(axis=1)
    out: list[FloatArray] = []
    for m in (masks, 1.0 - masks):
        sum1, sum2 = m @ s1, m @ s2
        mean = sum1 / n_rows + centre
        if metric == "mean":
            out.append(mean)
            continue
        if metric == "sharpe":
            var = np.maximum(sum2 - sum1**2 / n_rows, 0.0) / (n_rows - 1)
            out.append(_ratio(mean, np.sqrt(var)))
        else:
            out.append(_ratio(mean, np.sqrt((m @ s_down) / n_rows)))
    return out[0], out[1]


def _callable_metric(
    blocks: FloatArray, masks: FloatArray, fn: MetricFunction
) -> tuple[FloatArray, FloatArray]:
    train: list[FloatArray] = []
    test: list[FloatArray] = []
    n = blocks.shape[2]
    for mask in masks.astype(bool):
        train.append(np.asarray(fn(blocks[mask].reshape(-1, n)), dtype=np.float64))
        test.append(np.asarray(fn(blocks[~mask].reshape(-1, n)), dtype=np.float64))
    return np.vstack(train), np.vstack(test)


def probability_of_backtest_overfitting(
    returns: npt.ArrayLike | pd.DataFrame,
    *,
    n_splits: int = 16,
    metric: Metric | MetricFunction = "sharpe",
) -> PBOResult:
    """Estimate the probability of backtest overfitting of a parameter search (CSCV).

    Args:
        returns: ``(T, N)`` per-period returns of ``N >= 2`` configurations over the same
            periods, e.g. :attr:`backtester.research.GridSearchResult.returns`.
        n_splits: Even number ``S`` of blocks; the procedure evaluates ``C(S, S/2)``
            combinations (12,870 for ``S = 16``).
        metric: Performance statistic used both to select and to rank: the per-period
            ``"sharpe"`` ratio (default), ``"mean"`` return, ``"sortino"`` ratio, or a
            function of a ``(rows, N)`` array (evaluated per combination, slower).

    Returns:
        A :class:`PBOResult`. Ties in the test ranking get average ranks; the training
        winner is the first column among equals.
    """
    values, labels = _as_matrix(returns)
    t, n = values.shape
    if n < 2:
        raise ValueError("need at least two configurations")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits must be an even number >= 2")
    rows = t // n_splits
    if rows < 2:
        raise ValueError(f"need at least {2 * n_splits} periods for {n_splits} splits")
    used = values[t - rows * n_splits :]
    blocks = used.reshape(n_splits, rows, n)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    masks = np.zeros((len(combos), n_splits))
    for c, chosen in enumerate(combos):
        masks[c, list(chosen)] = 1.0
    if callable(metric):
        train, test = _callable_metric(blocks, masks, metric)
        metric_name = getattr(metric, "__name__", "custom")
    else:
        if metric not in ("sharpe", "mean", "sortino"):
            raise ValueError(f"unknown metric {metric!r}")
        train, test = _block_metric(blocks, masks, metric)
        metric_name = metric
    # NaN performance (possible only for custom metrics) never wins and ranks last.
    selected = np.argmax(np.where(np.isnan(train), -np.inf, train), axis=1)
    test_ranked = np.where(np.isnan(test), -np.inf, test)
    idx = np.arange(len(combos))
    chosen_oos = test_ranked[idx, selected]
    below = (test_ranked < chosen_oos[:, None]).sum(axis=1)
    ties = (test_ranked == chosen_oos[:, None]).sum(axis=1)  # includes itself
    rank = below + (ties + 1) / 2.0
    w = rank / (n + 1)
    logits = np.log(w / (1.0 - w))
    is_perf = train[idx, selected]
    oos_perf = test[idx, selected]
    finite = np.isfinite(is_perf) & np.isfinite(oos_perf)
    slope = intercept = r2 = math.nan
    if finite.sum() > 2 and np.ptp(is_perf[finite]) > 0:
        slope, intercept = (float(v) for v in np.polyfit(is_perf[finite], oos_perf[finite], 1))
        corr = np.corrcoef(is_perf[finite], oos_perf[finite])[0, 1]
        r2 = float(corr**2)
    return PBOResult(
        pbo=float(np.mean(logits <= 0)),
        logits=logits,
        relative_ranks=w,
        is_performance=is_perf,
        oos_performance=oos_perf,
        selected=selected.astype(np.intp),
        degradation_slope=slope,
        degradation_intercept=intercept,
        degradation_r2=r2,
        prob_oos_loss=float(np.mean(oos_perf < 0)),
        n_splits=n_splits,
        n_obs=rows * n_splits,
        labels=labels,
        metric=metric_name,
    )
