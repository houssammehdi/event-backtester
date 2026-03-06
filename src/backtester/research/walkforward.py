"""Walk-forward optimisation: rolling in-sample fits, out-of-sample tests, stitched OOS."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtester.analytics.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from backtester.config import BacktestConfig
from backtester.data.feed import DataFeed
from backtester.engine import BacktestResult
from backtester.errors import ConfigError
from backtester.research.grid import (
    MULTIPLE_TESTING_NOTE,
    GridSearchResult,
    Objective,
    _rank,
    _warmup_bars,
    expand_grid,
    sharpe_objective,
)
from backtester.research.parallel import RunSpec, StrategyFactory, backtest_runner
from backtester.validation.pbo import probability_of_backtest_overfitting

if TYPE_CHECKING:
    from backtester.validation.report import StrategyValidation


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """Calendar positions (inclusive) of one in-sample / out-of-sample split."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def __post_init__(self) -> None:
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ConfigError(f"invalid walk-forward window {self}")


def walk_forward_windows(
    n_bars: int,
    train_size: int,
    test_size: int,
    *,
    step: int | None = None,
    anchored: bool = False,
    gap: int = 0,
    start: int = 0,
    allow_partial: bool = True,
) -> list[WalkForwardWindow]:
    """Generate walk-forward windows over a calendar of ``n_bars`` bars.

    Args:
        n_bars: Length of the calendar.
        train_size: In-sample length in bars (the initial length when ``anchored``).
        test_size: Out-of-sample length in bars.
        step: Shift between consecutive windows; defaults to ``test_size`` so the test
            windows tile the timeline. Must be ``>= test_size`` so OOS periods never
            overlap (overlap would double count when stitching).
        anchored: Keep the in-sample start fixed (expanding window) instead of rolling.
        gap: Bars skipped between the end of training and the start of testing (an
            embargo against leakage from overlapping signal horizons).
        start: Calendar position of the first training bar (bars before it are
            available only as warm-up history).
        allow_partial: Keep a final, shorter test window that runs to the last bar.

    Returns:
        Windows in chronological order.
    """
    step = test_size if step is None else step
    if min(train_size, test_size, step) < 1 or gap < 0 or start < 0:
        raise ConfigError("train_size, test_size and step must be >= 1; gap, start >= 0")
    if step < test_size:
        raise ConfigError("step < test_size would make out-of-sample windows overlap")
    windows: list[WalkForwardWindow] = []
    offset = 0
    while True:
        train_start = start if anchored else start + offset
        train_end = start + offset + train_size - 1
        test_start = train_end + 1 + gap
        if test_start >= n_bars:
            break
        test_end = test_start + test_size - 1
        if test_end >= n_bars:
            if not allow_partial:
                break
            test_end = n_bars - 1
        windows.append(WalkForwardWindow(train_start, train_end, test_start, test_end))
        offset += step
    if not windows:
        raise ConfigError("calendar too short for a single walk-forward window")
    return windows


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Outcome of :func:`walk_forward`."""

    windows: pd.DataFrame
    """One row per window: bounds, chosen parameters, in-sample and OOS scores, DSR."""
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_results: list[BacktestResult]
    initial_cash: float
    periods_per_year: float
    n_trials: int

    def metrics(self) -> dict[str, float]:
        """Headline statistics of the stitched out-of-sample equity curve."""
        eq = pd.concat(
            [pd.Series([self.initial_cash]), self.oos_equity.reset_index(drop=True)],
            ignore_index=True,
        )
        ppy = self.periods_per_year
        is_mean = float(self.windows["is_sharpe"].mean())
        oos_sharpe = sharpe_ratio(self.oos_returns, ppy)
        return {
            "total_return": total_return(eq),
            "cagr": cagr(eq, ppy),
            "annual_volatility": annualized_volatility(self.oos_returns, ppy),
            "sharpe": oos_sharpe,
            "max_drawdown": max_drawdown(eq),
            "mean_is_sharpe": is_mean,
            "walk_forward_efficiency": oos_sharpe / is_mean if is_mean > 0 else math.nan,
        }

    def note(self) -> str:
        """Multiple-testing warning for the in-sample searches."""
        return MULTIPLE_TESTING_NOTE.format(n=self.n_trials)

    def validate(self, **kwargs: Any) -> StrategyValidation:
        """Bootstrap intervals, PSR and MinTRL of the stitched out-of-sample returns.

        The stitched curve is out of sample, so no deflation for the in-sample search
        applies to it. Keyword arguments go to
        :func:`~backtester.validation.validate_returns`.
        """
        from backtester.validation.report import validate_returns

        options: dict[str, Any] = {"periods_per_year": self.periods_per_year}
        options.update(kwargs)
        return validate_returns(self.oos_returns, **options)


def _in_sample_pbo(search: GridSearchResult, n_splits: int) -> float:
    """PBO of one window's in-sample search (NaN for a single configuration)."""
    splits = min(n_splits, len(search.returns) // 2)
    splits -= splits % 2
    if search.returns.shape[1] < 2 or splits < 2:
        return math.nan
    return probability_of_backtest_overfitting(search.returns, n_splits=splits).pbo


def walk_forward(
    feed: DataFeed,
    factory: StrategyFactory,
    grid: Mapping[str, Sequence[Any]],
    windows: Sequence[WalkForwardWindow],
    *,
    config: BacktestConfig | None = None,
    objective: Objective = sharpe_objective,
    fixed: Mapping[str, Any] | None = None,
    pbo_splits: int = 8,
    n_jobs: int = 1,
) -> WalkForwardResult:
    """Run a walk-forward optimisation.

    For each window the grid is searched on the in-sample span using a feed that is
    physically truncated at ``train_end`` - the optimiser cannot see later data. The
    winning parameters are then run on the out-of-sample span with a feed truncated at
    ``test_end``; bars before ``test_start`` are replayed as warm-up only (the
    strategy starts flat, with its intents dropped until ``test_start``).

    Out-of-sample returns are stitched into one equity curve by compounding. Every
    window starts from flat, so the stitched curve includes re-entry costs at each
    boundary.

    Each row of :attr:`WalkForwardResult.windows` reports the in-sample deflated
    Sharpe ratio (``is_dsr``) and the probability of backtest overfitting of the
    in-sample search (``is_pbo``, CSCV with ``pbo_splits`` blocks).

    With ``n_jobs > 1`` (``-1``: one per CPU) the in-sample runs of all windows, then
    the out-of-sample runs, execute in worker processes; the result is identical to
    the serial one (see :mod:`backtester.research.parallel`).
    """
    if not windows:
        raise ConfigError("need at least one window")
    for before, after in itertools.pairwise(windows):
        if after.test_start <= before.test_end:
            raise ConfigError("out-of-sample windows overlap")
    cfg = config or BacktestConfig()
    combos = expand_grid(grid)
    base = dict(fixed or {})
    lookback = _warmup_bars(factory, [RunSpec({**base, **p}) for p in combos])
    train = [
        RunSpec({**base, **p}, w.train_start, w.train_end, until=w.train_end)
        for w in windows
        for p in combos
    ]
    with backtest_runner(feed, factory, cfg, n_jobs=n_jobs) as run:
        runs = run(train)
        searches = [
            _rank(combos, [next(runs) for _ in combos], objective, lookback) for _ in windows
        ]
        test = [
            RunSpec({**base, **s.best_params}, w.test_start, w.test_end, until=w.test_end)
            for w, s in zip(windows, searches, strict=True)
        ]
        oos_results = list(run(test))
    rows: list[dict[str, Any]] = []
    oos_parts: list[pd.Series] = []
    for k, (w, search, oos) in enumerate(zip(windows, searches, oos_results, strict=True)):
        previous_equity = oos.equity.shift(1)
        previous_equity.iloc[0] = oos.initial_cash
        oos_parts.append(oos.equity / previous_equity - 1.0)
        index = feed.index
        rows.append(
            {
                "window": k,
                "train_start": index[w.train_start],
                "train_end": index[w.train_end],
                "test_start": index[w.test_start],
                "test_end": index[w.test_end],
                **{f"param_{p}": v for p, v in search.best_params.items()},
                "is_objective": float(search.table["objective"].iloc[0]),
                "is_sharpe": float(search.table["sharpe"].iloc[0]),
                "is_dsr": search.deflated_sharpe,
                "is_pbo": _in_sample_pbo(search, pbo_splits),
                "oos_sharpe": sharpe_ratio(oos.returns, oos.periods_per_year),
                "oos_return": total_return(
                    pd.concat([pd.Series([oos.initial_cash]), oos.equity.reset_index(drop=True)])
                ),
            }
        )
    returns = pd.concat(oos_parts).rename("oos_returns")
    equity = (cfg.initial_cash * (1.0 + returns).cumprod()).rename("oos_equity")
    return WalkForwardResult(
        windows=pd.DataFrame(rows),
        oos_returns=returns,
        oos_equity=equity,
        oos_results=oos_results,
        initial_cash=cfg.initial_cash,
        periods_per_year=cfg.periods_per_year,
        n_trials=len(combos),
    )
