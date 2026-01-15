from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from backtester import ConfigError, DataFeed, MarketView, StrategyContext, generate_ohlcv
from backtester.research import (
    WalkForwardWindow,
    deflated_sharpe_ratio,
    expand_grid,
    expected_max_sharpe,
    grid_search,
    probabilistic_sharpe_ratio,
    sample_moments,
    walk_forward,
    walk_forward_windows,
)
from backtester.strategies import SmaCrossover


@pytest.fixture(scope="module")
def feed() -> DataFeed:
    return DataFeed(generate_ohlcv(n_symbols=3, years=4, seed=3))


def test_expand_grid() -> None:
    combos = expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert combos == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    assert expand_grid({}) == [{}]
    with pytest.raises(ConfigError):
        expand_grid({"a": []})


def test_grid_search_ranks_by_objective(feed: DataFeed) -> None:
    grid = {"fast": [5, 20], "slow": [50, 100]}
    search = grid_search(feed, SmaCrossover, grid)
    table = search.table
    assert search.n_trials == 4
    assert list(table["objective"]) == sorted(table["objective"], reverse=True)
    assert search.best_params == {k: table.iloc[0][k] for k in grid}
    assert search.best_result.params["fast"] == search.best_params["fast"]
    assert 0.0 <= search.deflated_sharpe <= 1.0
    assert "4 configurations" in search.note()


class TestWindows:
    def test_rolling_windows_tile_the_timeline(self) -> None:
        w = walk_forward_windows(1000, train_size=300, test_size=100, start=50)
        assert w[0] == WalkForwardWindow(50, 349, 350, 449)
        assert all(b.test_start == a.test_end + 1 for a, b in itertools.pairwise(w))
        assert all(x.train_end - x.train_start == 299 for x in w)
        assert w[-1].test_end == 999  # partial final window kept

    def test_anchored_gap_and_no_partial(self) -> None:
        w = walk_forward_windows(1000, 300, 100, anchored=True, gap=5, allow_partial=False)
        assert all(x.train_start == 0 for x in w)
        assert all(x.test_start == x.train_end + 6 for x in w)
        assert all(x.test_end - x.test_start == 99 for x in w)

    def test_invalid(self) -> None:
        with pytest.raises(ConfigError, match="overlap"):
            walk_forward_windows(1000, 300, 100, step=50)
        with pytest.raises(ConfigError, match="too short"):
            walk_forward_windows(100, 300, 100)
        with pytest.raises(ConfigError):
            WalkForwardWindow(0, 10, 10, 20)

    @given(
        n=st.integers(50, 3000),
        train=st.integers(1, 800),
        test=st.integers(1, 400),
        extra_step=st.integers(0, 100),
        gap=st.integers(0, 20),
        anchored=st.booleans(),
    )
    def test_windows_never_overlap_or_leak(
        self, n: int, train: int, test: int, extra_step: int, gap: int, anchored: bool
    ) -> None:
        try:
            windows = walk_forward_windows(
                n, train, test, step=test + extra_step, gap=gap, anchored=anchored
            )
        except ConfigError:
            assert train + gap >= n
            return
        for w in windows:
            assert 0 <= w.train_start <= w.train_end < w.test_start <= w.test_end < n
            assert w.test_start - w.train_end - 1 == gap
        for a, b in itertools.pairwise(windows):
            assert b.test_start > a.test_end  # OOS windows are disjoint and ordered
            assert b.train_end > a.train_end


class Spy(SmaCrossover):
    """Records the latest bar any instance was shown, per phase."""

    seen: ClassVar[list[tuple[int, pd.Timestamp]]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run_id = len(Spy.seen)
        Spy.seen.append((self.run_id, pd.Timestamp.min))

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        Spy.seen[self.run_id] = (self.run_id, view.timestamp)
        super().on_bar(view, ctx)


def test_walk_forward_fits_never_see_out_of_sample_data(feed: DataFeed) -> None:
    Spy.seen.clear()
    grid = {"fast": [5, 10], "slow": [40, 80]}
    windows = walk_forward_windows(len(feed), 252, 126, start=100, gap=3)
    wf = walk_forward(feed, Spy, grid, windows)
    per_window = len(expand_grid(grid)) + 1  # fits, then one OOS run
    assert len(Spy.seen) == per_window * len(windows)
    for k, w in enumerate(windows):
        runs = Spy.seen[k * per_window : (k + 1) * per_window]
        for _, last_seen in runs[:-1]:
            assert last_seen == feed.index[w.train_end]  # fitting stops at the IS end
        assert runs[-1][1] == feed.index[w.test_end]
    # stitched OOS covers exactly the test windows, once each
    expected = sum(w.test_end - w.test_start + 1 for w in windows)
    assert len(wf.oos_returns) == expected
    assert wf.oos_returns.index.is_unique
    assert wf.oos_returns.index.is_monotonic_increasing
    np.testing.assert_allclose(
        wf.oos_equity.to_numpy(), 1_000_000 * np.cumprod(1 + wf.oos_returns.to_numpy())
    )
    m = wf.metrics()
    assert set(m) >= {"sharpe", "cagr", "max_drawdown", "walk_forward_efficiency"}
    assert list(wf.windows["test_start"]) == [feed.index[w.test_start] for w in windows]
    assert "configurations" in wf.note()


def test_walk_forward_rejects_overlapping_windows(feed: DataFeed) -> None:
    bad = [WalkForwardWindow(0, 99, 100, 199), WalkForwardWindow(50, 149, 150, 249)]
    with pytest.raises(ConfigError, match="overlap"):
        walk_forward(feed, SmaCrossover, {"fast": [5]}, bad)
    with pytest.raises(ConfigError):
        walk_forward(feed, SmaCrossover, {"fast": [5]}, [])


class TestDeflatedSharpe:
    def test_psr_hand_value(self) -> None:
        sr, n = 0.1, 253
        expected = NormalDist().cdf(sr * math.sqrt(n - 1) / math.sqrt(1 + 0.5 * sr**2))
        assert probabilistic_sharpe_ratio(sr, 0.0, n) == pytest.approx(expected)
        assert probabilistic_sharpe_ratio(0.05, 0.05, 100) == pytest.approx(0.5)
        # negative skew and fat tails widen the uncertainty
        assert probabilistic_sharpe_ratio(0.1, 0.0, n, skew=-1, kurtosis=8) < expected

    def test_expected_max_grows_with_trials(self) -> None:
        values = [expected_max_sharpe(n, 0.01) for n in (1, 2, 10, 100, 1000)]
        assert values[0] == 0.0
        assert all(b > a for a, b in itertools.pairwise(values))
        # ~ sqrt(2 ln N) scaling for large N
        assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.25, abs=0.05)

    def test_deflation_penalises_many_trials(self) -> None:
        rng = np.random.default_rng(5)
        returns = rng.normal(0.0008, 0.01, 1000)
        sr, skew, kurt, n = sample_moments(returns)
        assert n == 1000
        assert sr == pytest.approx(returns.mean() / returns.std(ddof=1))
        single = deflated_sharpe_ratio(returns, [sr])
        assert single == pytest.approx(probabilistic_sharpe_ratio(sr, 0, n, skew, kurt))
        many = deflated_sharpe_ratio(returns, rng.normal(0, 0.03, 200))
        assert many < single

    def test_degenerate_inputs(self) -> None:
        assert math.isnan(sample_moments([0.01, 0.01, 0.01])[0])
        assert math.isnan(probabilistic_sharpe_ratio(0.1, 0.0, 1))
