"""Cross-validation for time series with overlapping information: purging and embargo.

Observation ``i`` of a financial data set usually depends on an interval of time, not
an instant: a label built from the next ``h`` returns, a feature computed over the last
``L`` bars, or a strategy return that depends on a position formed from a look-back
window. Standard K-fold cross-validation then leaks, because training observations
whose intervals overlap the test period carry information about it (Lopez de Prado,
2018, ch. 7). Two fixes:

* **Purging** removes from the training set every observation whose interval
  ``[start_i, end_i]`` overlaps the interval spanned by a contiguous block of test
  observations (closed intervals: touching counts as overlapping).
* **Embargo** also removes the first ``embargo`` observations after each test block,
  against leakage through serial correlation that interval overlap does not capture.

:class:`PurgedKFold` applies them to ``K`` contiguous folds. :class:`CombinatorialPurgedCV`
(CPCV, Lopez de Prado, 2018, ch. 12) splits the data into ``N`` groups and uses every
choice of ``k`` of them as the test set. Each group is then tested ``C(N-1, k-1)``
times, which lets the out-of-sample predictions be stitched into that many complete
*backtest paths* instead of the single path of walk-forward testing.
:func:`cpcv_backtest` applies CPCV to parameter selection: in every split it selects the
configuration with the best training performance and records its test returns.

Reference:
    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
    Chapters 7 and 12.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class CVSplit:
    """Integer positions of one train/test split."""

    train: IntArray
    test: IntArray
    test_groups: tuple[int, ...]
    """Indices of the groups (folds) that form the test set."""


def label_spans(n_obs: int, *, lookback: int = 0, horizon: int = 0) -> tuple[IntArray, IntArray]:
    """Information intervals ``[i - lookback, i + horizon]`` for positions ``i = 0..n-1``.

    Use ``horizon`` for labels built from future bars (a ``h``-bar forward return) and
    ``lookback`` for observations that depend on past bars (a strategy return whose
    position comes from an ``L``-bar signal).
    """
    if n_obs < 1 or lookback < 0 or horizon < 0:
        raise ValueError("n_obs must be positive and lookback, horizon non-negative")
    positions = np.arange(n_obs, dtype=np.intp)
    return positions - lookback, positions + horizon


def _spans(
    start: npt.ArrayLike, end: npt.ArrayLike | None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # integer positions, floats or datetime64 values: any ordered dtype
    s = np.asarray(start)
    e = s if end is None else np.asarray(end)
    if s.ndim != 1 or s.shape != e.shape:
        raise ValueError("start and end must be 1-D arrays of the same length")
    if len(s) and (np.any(s[1:] < s[:-1]) or np.any(e < s)):
        raise ValueError("start must be sorted and every end >= its start")
    return s, e


def _blocks(test: IntArray) -> list[IntArray]:
    """Split sorted test positions into runs of consecutive positions."""
    if len(test) == 0:
        return []
    cuts = np.flatnonzero(np.diff(test) > 1) + 1
    return list(np.split(test, cuts))


def purged_train_indices(
    start: npt.ArrayLike,
    end: npt.ArrayLike | None,
    test: npt.ArrayLike,
    *,
    embargo: int = 0,
) -> IntArray:
    """Training positions left after purging and embargoing around ``test``.

    Args:
        start: Start of each observation's information interval, sorted.
        end: End of each interval (``None``: point intervals, ``end = start``).
        test: Positions of the test observations.
        embargo: Observations dropped right after each contiguous test block (on top
            of those purged for overlapping it).
    """
    s, e = _spans(start, end)
    n = len(s)
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    test_idx = np.unique(np.asarray(test, dtype=np.intp))
    keep = np.ones(n, dtype=bool)
    keep[test_idx] = False
    for block in _blocks(test_idx):
        lo, hi = s[block[0]], np.max(e[block])
        keep &= ~((s <= hi) & (e >= lo))  # purge: closed-interval overlap
        if embargo:
            after = int(np.searchsorted(s, hi, side="right"))  # first start beyond the block
            keep[after : after + embargo] = False
    out: IntArray = np.flatnonzero(keep).astype(np.intp)
    return out


class PurgedKFold:
    """K contiguous folds with purging and embargo.

    Args:
        n_splits: Number of folds ``K``.
        embargo: Observations embargoed after each test fold.
    """

    def __init__(self, n_splits: int = 5, *, embargo: int = 0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, start: npt.ArrayLike, end: npt.ArrayLike | None = None) -> list[CVSplit]:
        """One split per fold, in order; see :func:`purged_train_indices`."""
        s, e = _spans(start, end)
        if len(s) < self.n_splits:
            raise ValueError("fewer observations than folds")
        folds = np.array_split(np.arange(len(s), dtype=np.intp), self.n_splits)
        return [
            CVSplit(
                train=purged_train_indices(s, e, fold, embargo=self.embargo),
                test=fold,
                test_groups=(k,),
            )
            for k, fold in enumerate(folds)
        ]


class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation (CPCV).

    The observations are cut into ``n_groups`` contiguous groups; each of the
    ``C(n_groups, n_test_groups)`` splits tests on one choice of ``n_test_groups`` of
    them and trains on the rest, purged and embargoed around every test block (adjacent
    test groups form one block).

    Args:
        n_groups: Number of groups ``N``.
        n_test_groups: Groups per test set ``k`` (``1 <= k < N``).
        embargo: Observations embargoed after each test block.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, *, embargo: int = 0) -> None:
        if n_groups < 2 or not 1 <= n_test_groups < n_groups:
            raise ValueError("need n_groups >= 2 and 1 <= n_test_groups < n_groups")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo

    @property
    def n_splits(self) -> int:
        """``C(N, k)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths, ``C(N - 1, k - 1) = k / N * C(N, k)``."""
        return math.comb(self.n_groups - 1, self.n_test_groups - 1)

    def combinations(self) -> list[tuple[int, ...]]:
        """Test groups of every split, in lexicographic order (the split order)."""
        return list(itertools.combinations(range(self.n_groups), self.n_test_groups))

    def groups(self, n_obs: int) -> list[IntArray]:
        """Positions in each group (near-equal contiguous chunks)."""
        if n_obs < self.n_groups:
            raise ValueError("fewer observations than groups")
        return list(np.array_split(np.arange(n_obs, dtype=np.intp), self.n_groups))

    def split(self, start: npt.ArrayLike, end: npt.ArrayLike | None = None) -> list[CVSplit]:
        """All ``C(N, k)`` splits in lexicographic order of their test groups."""
        s, e = _spans(start, end)
        groups = self.groups(len(s))
        out = []
        for chosen in self.combinations():
            test = np.concatenate([groups[g] for g in chosen])
            train = purged_train_indices(s, e, test, embargo=self.embargo)
            out.append(CVSplit(train=train, test=test, test_groups=chosen))
        return out

    def path_splits(self) -> IntArray:
        """Which split feeds each group of each backtest path.

        Returns an ``(n_paths, n_groups)`` array of split indices. Group ``g`` is tested
        by ``n_paths`` splits; path ``p`` takes the ``p``-th of them in split order, so
        every split contributes each of its test groups to exactly one path.
        """
        table = np.empty((self.n_paths, self.n_groups), dtype=np.intp)
        for g in range(self.n_groups):
            testing = [s for s, chosen in enumerate(self.combinations()) if g in chosen]
            table[:, g] = testing
        return table

    def assemble_paths(self, n_obs: int, predictions: Sequence[npt.ArrayLike]) -> FloatArray:
        """Stitch per-split out-of-sample values into complete paths.

        Args:
            n_obs: Number of observations the splits were made for.
            predictions: One array per split (in :meth:`split` order), aligned with
                that split's ``test`` positions.

        Returns:
            ``(n_paths, n_obs)`` array; row ``p`` is backtest path ``p``.
        """
        if len(predictions) != self.n_splits:
            raise ValueError(f"expected {self.n_splits} prediction arrays")
        groups = self.groups(n_obs)
        combos = self.combinations()
        paths = np.full((self.n_paths, n_obs), np.nan)
        for p, row in enumerate(self.path_splits()):
            for g, s in enumerate(row):
                values = np.asarray(predictions[s], dtype=np.float64)
                offset = 0
                for other in combos[s]:
                    if other == g:
                        break
                    offset += len(groups[other])
                paths[p, groups[g]] = values[offset : offset + len(groups[g])]
        return paths


@dataclass(frozen=True, slots=True)
class CPCVResult:
    """Output of :func:`cpcv_backtest`."""

    path_returns: pd.DataFrame = field(repr=False)
    """Out-of-sample returns of each backtest path (columns ``path_0``, ...)."""
    path_sharpe: FloatArray
    """Annualised Sharpe ratio of every path."""
    selected: tuple[str, ...]
    """Configuration chosen in each split."""
    n_groups: int
    n_test_groups: int
    embargo: int
    periods_per_year: float

    @property
    def n_paths(self) -> int:
        """Number of backtest paths."""
        return len(self.path_sharpe)


def _sharpe_columns(x: FloatArray) -> FloatArray:
    """Per-period Sharpe ratio of each column (0/0 := 0, x/0 := sign(x) * inf)."""
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    flat = np.where(mean == 0, 0.0, np.sign(mean) * np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: FloatArray = np.where(sd > 0, mean / sd, flat)
    return out


def cpcv_backtest(
    returns: pd.DataFrame,
    cv: CombinatorialPurgedCV | None = None,
    *,
    start: npt.ArrayLike | None = None,
    end: npt.ArrayLike | None = None,
    metric: Literal["sharpe", "mean"] = "sharpe",
    periods_per_year: float = 252.0,
) -> CPCVResult:
    """Parameter selection evaluated with combinatorial purged cross-validation.

    In every CPCV split the configuration (column of ``returns``) with the best
    training performance is selected, and its returns over the test groups become that
    split's out-of-sample predictions; they are then stitched into backtest paths.

    Args:
        returns: ``(T, N)`` per-period returns of ``N`` configurations over the same
            periods, each run continuously over the whole sample (so returns at bar
            ``t`` use data up to ``t`` only).
        cv: The splitter (default: 6 groups, 2 test groups, no embargo).
        start: Start of each observation's information interval (default: the
            observation's own position). Use :func:`label_spans` with the strategies'
            look-back to purge returns that depend on test-period prices.
        end: End of each interval (default: ``start``).
        metric: Training-set score: per-period ``"sharpe"`` ratio or ``"mean"`` return.
        periods_per_year: Annualisation of the path Sharpe ratios.
    """
    splitter = cv or CombinatorialPurgedCV()
    values = returns.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("returns must be a (periods x configurations) DataFrame")
    if np.isnan(values).any():
        raise ValueError("returns contain NaN")
    n = values.shape[0]
    s = np.arange(n, dtype=np.intp) if start is None else np.asarray(start)
    splits = splitter.split(s, end)
    labels = tuple(str(c) for c in returns.columns)
    predictions: list[FloatArray] = []
    selected: list[str] = []
    for split in splits:
        train = values[split.train]
        if len(train) < 3:
            raise ValueError("a training set has fewer than 3 observations; reduce embargo")
        score = _sharpe_columns(train) if metric == "sharpe" else train.mean(axis=0)
        best = int(np.argmax(np.where(np.isnan(score), -np.inf, score)))
        selected.append(labels[best])
        predictions.append(values[split.test, best])
    paths = splitter.assemble_paths(n, predictions)
    frame = pd.DataFrame(
        paths.T, index=returns.index, columns=[f"path_{p}" for p in range(len(paths))]
    )
    sharpe = _sharpe_columns(paths.T) * math.sqrt(periods_per_year)
    return CPCVResult(
        path_returns=frame,
        path_sharpe=sharpe,
        selected=tuple(selected),
        n_groups=splitter.n_groups,
        n_test_groups=splitter.n_test_groups,
        embargo=splitter.embargo,
        periods_per_year=periods_per_year,
    )
