"""Batches of independent backtests, optionally spread over worker processes.

Grid searches and walk-forward analyses run many backtests that share nothing but
their inputs. :func:`backtest_runner` runs such a batch either in this process or in a
``concurrent.futures`` process pool. Parallel runs return the same results, in the
same order, as the serial loop: every run builds its own strategy, broker and risk
manager from the same pickled inputs, so nothing depends on which worker ran it or
when.

With ``n_jobs > 1`` the strategy factory must be picklable (a class, or a function
defined at module level; not a lambda), and so must the config. Workers are started
with the ``forkserver`` method where available (``spawn`` elsewhere), so scripts that
use them need the usual ``if __name__ == "__main__":`` guard.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from backtester.config import BacktestConfig, Bound
from backtester.data.feed import DataFeed
from backtester.engine import BacktestResult
from backtester.errors import ConfigError
from backtester.strategy.base import Strategy

StrategyFactory = Callable[..., Strategy]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One backtest of a batch."""

    params: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments for the strategy factory."""
    start: Bound = None
    """First bar of the trading window (earlier bars are warm-up)."""
    end: Bound = None
    """Last bar of the trading window."""
    until: int | None = None
    """If set, the run gets a copy of the feed that ends at this calendar position, so
    later data does not exist for it (as in walk-forward analysis)."""


Runner = Callable[[Iterable[RunSpec]], Iterator[BacktestResult]]


def available_cpus() -> int:
    """CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on macOS and Windows
        return os.cpu_count() or 1


def resolve_jobs(n_jobs: int) -> int:
    """Worker count for ``n_jobs``: a positive count, or ``-1`` for every available CPU."""
    if n_jobs == -1:
        return available_cpus()
    if n_jobs < 1:
        raise ConfigError(f"n_jobs must be a positive integer or -1 (all CPUs), got {n_jobs}")
    return n_jobs


class _Batch:
    """Runs specs against one feed, keeping the most recent truncated copy of it."""

    def __init__(self, feed: DataFeed, factory: StrategyFactory, config: BacktestConfig) -> None:
        self.feed = feed
        self.factory = factory
        self.config = config
        self._truncated: tuple[int, DataFeed] | None = None

    def _feed_for(self, spec: RunSpec) -> DataFeed:
        if spec.until is None:
            return self.feed
        if self._truncated is None or self._truncated[0] != spec.until:
            self._truncated = (spec.until, self.feed.slice(0, spec.until + 1))
        return self._truncated[1]

    def __call__(self, spec: RunSpec) -> BacktestResult:
        strategy = self.factory(**spec.params)
        return self.config.run(self._feed_for(spec), strategy, start=spec.start, end=spec.end)


_WORKER_BATCH: _Batch | None = None


def _start_worker(feed: DataFeed, factory: StrategyFactory, config: BacktestConfig) -> None:
    global _WORKER_BATCH  # one batch per worker process
    _WORKER_BATCH = _Batch(feed, factory, config)


def _run_in_worker(spec: RunSpec) -> BacktestResult:
    if _WORKER_BATCH is None:  # pragma: no cover - the pool initializer always runs first
        raise RuntimeError("worker process was not initialised")
    return _WORKER_BATCH(spec)


def _start_method() -> str:
    methods = multiprocessing.get_all_start_methods()
    return "forkserver" if "forkserver" in methods else "spawn"


@contextmanager
def backtest_runner(
    feed: DataFeed,
    factory: StrategyFactory,
    config: BacktestConfig,
    *,
    n_jobs: int = 1,
) -> Iterator[Runner]:
    """Context manager yielding a function that runs a batch of :class:`RunSpec`.

    The function returns an iterator of results in spec order. With ``n_jobs == 1``
    the runs happen lazily in this process as the iterator is consumed; otherwise a
    pool of ``n_jobs`` worker processes (``-1``: one per CPU) runs them, receiving
    ``feed``, ``factory`` and ``config`` once per worker. The pool lives until the
    ``with`` block ends, so several batches can share it.
    """
    jobs = resolve_jobs(n_jobs)
    if jobs == 1:
        batch = _Batch(feed, factory, config)
        yield lambda specs: map(batch, specs)
        return
    for name, value in (("strategy factory", factory), ("config", config)):
        try:
            pickle.dumps(value)
        except (pickle.PicklingError, AttributeError, TypeError) as exc:
            raise ConfigError(
                f"n_jobs > 1 needs a picklable {name} (a class or a module-level "
                f"function, not a lambda or a local function): {exc}"
            ) from exc
    context = multiprocessing.get_context(_start_method())
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_start_worker,
        initargs=(feed, factory, config),
    ) as pool:
        yield lambda specs: pool.map(_run_in_worker, specs)
