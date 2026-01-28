"""Stationary bootstrap and Politis-White block-length selection."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester.analytics import cagr, max_drawdown, sharpe_ratio, sortino_ratio
from backtester.validation import (
    bootstrap_statistics,
    default_statistics,
    optimal_block_length,
    resample_counts,
    stationary_bootstrap_indices,
)


def ar1(phi: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 200)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


class TestBlockLength:
    def test_ar1_matches_the_theoretical_optimum(self) -> None:
        """For AR(1), b_SB = (2 phi / (1 - phi^2))^(2/3) n^(1/3) (flat-top window)."""
        n = 20_000
        for phi, seed in ((0.5, 1), (0.8, 2)):
            theory = (2 * phi / (1 - phi**2)) ** (2 / 3) * n ** (1 / 3)
            est = optimal_block_length(ar1(phi, n, seed))
            assert est.stationary == pytest.approx(theory, rel=0.3)
            assert est.circular == pytest.approx(est.stationary * 1.5 ** (1 / 3))

    def test_more_dependence_means_longer_blocks(self) -> None:
        phis = (0.0, 0.3, 0.6, 0.9)
        lengths = [optimal_block_length(ar1(phi, 5000, 3)).stationary for phi in phis]
        assert lengths == sorted(lengths)
        assert lengths[0] < 3  # iid: (almost) single observations

    @settings(max_examples=60, deadline=None)
    @given(
        n=st.integers(12, 1500),
        kind=st.sampled_from(["noise", "walk", "sine", "ar"]),
        seed=st.integers(0, 10_000),
    )
    def test_always_between_one_and_the_cap(self, n: int, kind: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n)
        series = {
            "noise": noise,
            "walk": np.cumsum(noise),
            "sine": np.sin(np.arange(n) / 7.0) + 0.1 * noise,
            "ar": ar1(0.7, n, seed),
        }[kind]
        est = optimal_block_length(series)
        cap = math.ceil(min(3 * math.sqrt(n), n / 3))
        assert 1.0 <= est.stationary <= cap
        assert est.stationary <= est.circular <= cap

    def test_degenerate_inputs(self) -> None:
        assert optimal_block_length(np.ones(50)).stationary == 1.0
        with pytest.raises(ValueError, match="12"):
            optimal_block_length(np.arange(5.0))


class TestIndices:
    def test_shape_range_and_determinism(self) -> None:
        idx = stationary_bootstrap_indices(100, 5.0, 50, seed=1)
        assert idx.shape == (50, 100)
        assert idx.min() >= 0
        assert idx.max() < 100
        np.testing.assert_array_equal(idx, stationary_bootstrap_indices(100, 5.0, 50, seed=1))

    def test_blocks_are_consecutive_and_geometric(self) -> None:
        b, n = 8.0, 2000
        idx = stationary_bootstrap_indices(n, b, 200, seed=2)
        continues = np.diff(idx, axis=1) % n == 1
        # a block continues with probability 1 - 1/b (random restarts land on i+1
        # with probability 1/n, negligible here)
        assert continues.mean() == pytest.approx(1 - 1 / b, abs=0.005)

    def test_block_length_one_is_the_iid_bootstrap(self) -> None:
        idx = stationary_bootstrap_indices(1000, 1.0, 20, seed=3)
        assert (np.diff(idx, axis=1) % 1000 == 1).mean() < 0.01

    def test_counts(self) -> None:
        idx = stationary_bootstrap_indices(30, 3.0, 7, seed=4)
        counts = resample_counts(idx, 30)
        assert counts.shape == (7, 30)
        np.testing.assert_array_equal(counts.sum(axis=1), 30)
        x = np.arange(30.0)
        np.testing.assert_allclose(counts @ x / 30, x[idx].mean(axis=1))

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="block_length"):
            stationary_bootstrap_indices(10, 0.5, 5)
        with pytest.raises(ValueError, match="positive"):
            stationary_bootstrap_indices(0, 2.0, 5)


class TestStatistics:
    def test_point_estimates_agree_with_the_metrics_module(self) -> None:
        rng = np.random.default_rng(5)
        r = rng.normal(0.0004, 0.012, 600)
        res = bootstrap_statistics(r, n_samples=50, seed=0)
        equity = np.r_[1.0, np.cumprod(1 + r)]
        assert res["sharpe"].estimate == pytest.approx(sharpe_ratio(r))
        assert res["sortino"].estimate == pytest.approx(sortino_ratio(r))
        assert res["cagr"].estimate == pytest.approx(cagr(equity))
        assert res["max_drawdown"].estimate == pytest.approx(max_drawdown(equity))
        assert res["total_return"].estimate == pytest.approx(equity[-1] - 1)

    def test_intervals_are_reproducible_and_ordered(self) -> None:
        r = np.random.default_rng(6).normal(0.0005, 0.01, 500)
        a = bootstrap_statistics(r, n_samples=300, seed=7)
        b = bootstrap_statistics(r, n_samples=300, seed=7)
        assert a.table().equals(b.table())
        for iv in a.intervals.values():
            assert iv.lower <= iv.upper
        assert len(a.samples["sharpe"]) == 300
        assert set(a.table().columns) == {"estimate", "lower", "upper", "std_error"}

    def test_basic_interval_reflects_percentiles(self) -> None:
        r = np.random.default_rng(8).normal(0.0005, 0.01, 400)
        pct = bootstrap_statistics(r, n_samples=400, seed=1, block_length=2.0)
        basic = bootstrap_statistics(r, n_samples=400, seed=1, block_length=2.0, method="basic")
        e = pct["sharpe"].estimate
        assert basic["sharpe"].lower == pytest.approx(2 * e - pct["sharpe"].upper)
        assert basic["sharpe"].upper == pytest.approx(2 * e - pct["sharpe"].lower)

    def test_dependence_widens_the_sharpe_interval(self) -> None:
        """Positively autocorrelated returns: the block bootstrap must not pretend iid."""
        r = 0.001 + 0.01 * ar1(0.6, 2000, 9) / 1.25
        iid = bootstrap_statistics(r, n_samples=400, block_length=1.0, seed=2)["sharpe"]
        auto = bootstrap_statistics(r, n_samples=400, seed=2)["sharpe"]
        assert auto.upper - auto.lower > 1.5 * (iid.upper - iid.lower)

    def test_custom_statistic_and_errors(self) -> None:
        r = np.random.default_rng(10).normal(0, 0.01, 100)
        res = bootstrap_statistics(r, {"median": lambda p: np.median(p, axis=1)}, n_samples=20)
        assert set(res.intervals) == {"median"}
        with pytest.raises(ValueError, match="method"):
            bootstrap_statistics(r, method="bca")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_statistics(r, confidence=0.0)
        with pytest.raises(ValueError, match="12"):
            bootstrap_statistics(r[:5])


def test_interval_coverage_on_iid_normal_data() -> None:
    """Percentile intervals cover the true mean, Sharpe and volatility near nominally."""
    rng = np.random.default_rng(2024)
    mu, sd, n, reps = 0.0005, 0.01, 250, 150
    truth = {"mean": mu, "sharpe": mu / sd * math.sqrt(252), "volatility": sd * math.sqrt(252)}
    stats = {k: v for k, v in default_statistics().items() if k in truth}
    hits = dict.fromkeys(truth, 0)
    for seed in range(reps):
        sample = rng.normal(mu, sd, n)
        res = bootstrap_statistics(sample, stats, n_samples=400, confidence=0.9, seed=seed)
        for k, value in truth.items():
            hits[k] += res[k].contains(value)
    for k in truth:
        assert hits[k] / reps == pytest.approx(0.9, abs=0.07), k


@pytest.mark.slow
def test_interval_coverage_large_simulation() -> None:
    rng = np.random.default_rng(7)
    mu, sd, n, reps = 0.0005, 0.01, 500, 1000
    truth = mu / sd * math.sqrt(252)
    stats = {"sharpe": default_statistics()["sharpe"]}
    hits = 0
    for seed in range(reps):
        sample = rng.normal(mu, sd, n)
        res = bootstrap_statistics(sample, stats, n_samples=500, confidence=0.9, seed=seed)
        hits += res["sharpe"].contains(truth)
    assert hits / reps == pytest.approx(0.9, abs=0.03)
