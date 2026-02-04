"""Exhaustive parameter grid search."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy.typing as npt
import pandas as pd

from backtester.analytics.metrics import (
    cagr,
    max_drawdown,
    sharpe_ratio,
    total_return,
)
from backtester.analytics.report import format_params
from backtester.config import BacktestConfig, Bound
from backtester.data.feed import DataFeed
from backtester.engine import BacktestResult
from backtester.errors import ConfigError
from backtester.strategy.base import Strategy
from backtester.strategy.rebalancing import TargetWeightStrategy
from backtester.validation.sharpe import deflated_sharpe_ratio, sample_moments

if TYPE_CHECKING:
    from backtester.validation.report import FamilyValidation

StrategyFactory = Callable[..., Strategy]
Objective = Callable[[BacktestResult], float]

MULTIPLE_TESTING_NOTE = (
    "Note: the best of {n} configurations was selected in-sample. Selection inflates "
    "performance; the deflated Sharpe ratio (DSR) is the probability that the chosen "
    "configuration's true Sharpe exceeds what the best of {n} skill-less trials would "
    "show by luck. Judge a strategy by out-of-sample (walk-forward) results, not by "
    "the in-sample optimum."
)


def sharpe_objective(result: BacktestResult) -> float:
    """Annualised Sharpe ratio of a run (``-inf`` when undefined)."""
    value = sharpe_ratio(result.returns, result.periods_per_year)
    return value if math.isfinite(value) else -math.inf


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid, in deterministic order."""
    if not grid:
        return [{}]
    keys = list(grid)
    for k in keys:
        if len(grid[k]) == 0:
            raise ConfigError(f"grid parameter {k!r} has no values")
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def config_label(params: Mapping[str, Any]) -> str:
    """Column label of one configuration: ``"k1=v1, k2=v2"`` (``"default"`` if empty)."""
    return format_params(dict(params)) or "default"


@dataclass(frozen=True, slots=True)
class GridSearchResult:
    """Outcome of :func:`grid_search`."""

    table: pd.DataFrame
    """One row per configuration, sorted by objective (best first)."""
    best_params: dict[str, Any]
    best_result: BacktestResult
    deflated_sharpe: float
    """Deflated Sharpe ratio of the best configuration given all trials."""
    returns: pd.DataFrame = field(repr=False)
    """Bar returns of every configuration over the evaluation window, one column per
    configuration in grid order (labels from :func:`config_label`): the performance
    matrix the overfitting and data-snooping statistics work on."""
    lookback: int = 0
    """Largest warm-up history any configuration needs (``history_bars`` of
    target-weight strategies; 0 when unknown)."""

    @property
    def n_trials(self) -> int:
        """Number of configurations evaluated."""
        return len(self.table)

    def note(self) -> str:
        """The multiple-testing warning for this search."""
        return MULTIPLE_TESTING_NOTE.format(n=self.n_trials)

    def validate(
        self,
        benchmark: pd.Series | npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> FamilyValidation:
        """Overfitting and data-snooping statistics of this search.

        Runs :func:`~backtester.validation.validate_family` on :attr:`returns`, with
        :attr:`lookback` as the purge interval of the CPCV. Keyword arguments are
        passed through (``n_samples``, ``pbo_splits``, ``benchmark_name``, ...).
        """
        from backtester.validation.report import validate_family

        options: dict[str, Any] = {
            "periods_per_year": self.best_result.periods_per_year,
            "lookback": self.lookback,
        }
        options.update(kwargs)
        return validate_family(self.returns, benchmark, **options)


def grid_search(
    feed: DataFeed,
    factory: StrategyFactory,
    grid: Mapping[str, Sequence[Any]],
    *,
    config: BacktestConfig | None = None,
    objective: Objective = sharpe_objective,
    start: Bound = None,
    end: Bound = None,
    fixed: Mapping[str, Any] | None = None,
) -> GridSearchResult:
    """Backtest every combination in ``grid`` and rank them by ``objective``.

    Args:
        feed: Data to run on. Only bars up to ``end`` are ever shown to strategies.
        factory: Callable building a fresh strategy from keyword parameters (typically
            the strategy class).
        grid: ``parameter -> candidate values``.
        config: Execution/risk settings (a fresh broker and risk manager per run).
        objective: Score to maximise; defaults to the annualised Sharpe ratio.
        start: First bar of the evaluation window (earlier bars are warm-up).
        end: Last bar of the evaluation window.
        fixed: Extra keyword arguments passed to every ``factory`` call.
    """
    cfg = config or BacktestConfig()
    combos = expand_grid(grid)
    rows: list[dict[str, Any]] = []
    results: list[BacktestResult] = []
    lookback = 0
    for params in combos:
        strategy = factory(**{**(fixed or {}), **params})
        if isinstance(strategy, TargetWeightStrategy):
            lookback = max(lookback, strategy.history_bars)
        result = cfg.run(feed, strategy, start=start, end=end)
        eq = result.equity
        per_period_sharpe = sample_moments(result.returns)[0]
        rows.append(
            {
                **params,
                "objective": objective(result),
                "sharpe": sharpe_ratio(result.returns, result.periods_per_year),
                "cagr": cagr(eq, result.periods_per_year),
                "total_return": total_return(eq),
                "max_drawdown": max_drawdown(eq),
                "n_fills": len(result.fills),
                "_sr": per_period_sharpe,
            }
        )
        results.append(result)
    table = pd.DataFrame(rows)
    order = table["objective"].sort_values(ascending=False, kind="stable").index
    best_pos = int(order[0])
    dsr = deflated_sharpe_ratio(results[best_pos].returns, table["_sr"].to_numpy())
    table = table.loc[order].drop(columns="_sr").reset_index(drop=True)
    returns = pd.DataFrame(
        {config_label(p): r.returns for p, r in zip(combos, results, strict=True)}
    )
    return GridSearchResult(
        table=table,
        best_params=dict(combos[best_pos]),
        best_result=results[best_pos],
        deflated_sharpe=dsr,
        returns=returns,
        lookback=lookback,
    )
