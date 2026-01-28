"""White's Reality Check and Hansen's SPA test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.validation import long_run_variance, superior_predictive_ability


def loop_variance(d: np.ndarray, block_length: float) -> np.ndarray:
    """Hansen (2005) / Politis & Romano (1994) kernel variance, written as a plain loop."""
    n = d.shape[0]
    q = 1.0 / block_length
    e = d - d.mean(axis=0)
    out = (e**2).sum(axis=0) / n
    for i in range(1, n):
        kappa = (1 - i / n) * (1 - q) ** i + (i / n) * (1 - q) ** (n - i)
        out = out + 2 * kappa * (e[: n - i] * e[i:]).sum(axis=0) / n
    return out


def test_long_run_variance_matches_the_direct_formula() -> None:
    rng = np.random.default_rng(0)
    d = rng.standard_normal((300, 4)).cumsum(axis=0) * 0.01 + rng.standard_normal((300, 4))
    for b in (1.0, 3.5, 20.0):
        np.testing.assert_allclose(long_run_variance(d, b), loop_variance(d, b), rtol=1e-10)
    # block length 1 is the iid bootstrap: the population variance
    np.testing.assert_allclose(long_run_variance(d, 1.0), d.var(axis=0), rtol=1e-12)


def test_pvalues_are_ordered_and_reproducible() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, (800, 12))
    r[:, 5] += 0.0004
    r[:, 7:] -= 0.001  # some poor strategies
    a = superior_predictive_ability(r, n_samples=500, seed=3)
    b = superior_predictive_ability(r, n_samples=500, seed=3)
    assert a.pvalue_lower <= a.pvalue_consistent <= a.pvalue_upper
    assert (a.pvalue_consistent, a.reality_check_pvalue) == (
        b.pvalue_consistent,
        b.reality_check_pvalue,
    )
    assert a.statistic == pytest.approx(max(a.t_statistics.max(), 0.0))
    assert a.best == int(np.argmax(a.t_statistics))


def test_benchmark_is_subtracted_and_labels_kept() -> None:
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0005, 0.01, 500)
    models = pd.DataFrame({"alpha": bench + 0.002 + rng.normal(0, 0.002, 500), "beta": bench})
    res = superior_predictive_ability(models, bench, n_samples=300)
    assert res.labels == ("alpha", "beta")
    assert res.best_label == "alpha"
    assert res.mean_differential[1] == 0.0  # identical to the benchmark
    assert res.pvalue_consistent < 0.01


def test_every_strategy_clearly_worse_gives_pvalue_one() -> None:
    """Regression: with T = 0, counting only strictly larger bootstrap values made the
    consistent and lower p-values ~0 for a family that trails the benchmark."""
    r = np.random.default_rng(3).normal(-0.002, 0.01, (1000, 5))
    res = superior_predictive_ability(r, n_samples=400)
    assert res.statistic == 0.0
    assert res.pvalue_lower == res.pvalue_consistent == res.pvalue_upper == 1.0
    assert res.reality_check_pvalue == 1.0


def test_a_skilled_strategy_is_detected() -> None:
    rng = np.random.default_rng(4)
    r = rng.normal(0.0, 0.01, (2000, 50))
    r[:, 17] += 0.0012  # per-period Sharpe 0.12 among 49 noise strategies (t ~ 5.4)
    res = superior_predictive_ability(r, n_samples=1000)
    assert res.pvalue_consistent < 0.01
    assert res.reality_check_pvalue < 0.01
    assert res.best == 17


def test_spa_keeps_power_when_poor_strategies_are_added() -> None:
    """Hansen's point, in two steps. 100 hopeless, volatile rivals swamp White's RC (the
    bootstrap max is driven by their large, unstudentised deviations). Studentising
    (SPA upper) removes the scale effect; also not recentring the hopeless ones
    (SPA consistent) restores the power to detect the one good strategy."""
    rng = np.random.default_rng(5)
    good = rng.normal(0.0008, 0.01, (1000, 1))
    poor = rng.normal(-0.003, 0.03, (1000, 100))
    res = superior_predictive_ability(np.hstack([good, poor]), n_samples=1000, seed=1)
    assert res.reality_check_pvalue > 0.9
    assert res.reality_check_pvalue > res.pvalue_upper > res.pvalue_consistent
    assert res.pvalue_consistent < 0.10


def test_degenerate_differentials() -> None:
    rng = np.random.default_rng(6)
    base = rng.normal(0, 0.01, 200)
    constant_winner = np.column_stack([base, base + 0.001])  # beats base by a constant
    res = superior_predictive_ability(constant_winner, base, n_samples=200)
    assert res.pvalue_consistent == 0.0
    identical = np.column_stack([base, base])
    res = superior_predictive_ability(identical, base, n_samples=200)
    assert res.pvalue_consistent == 1.0
    with pytest.raises(ValueError, match="periods"):
        superior_predictive_ability(np.zeros((100, 2)), np.zeros(50))
    with pytest.raises(ValueError, match="NaN"):
        superior_predictive_ability(np.full((100, 2), np.nan))


def _null_rejection_rates(n_sims: int, n_obs: int, n_models: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    p: dict[str, list[float]] = {"rc": [], "consistent": [], "upper": []}
    for s in range(n_sims):
        r = rng.standard_normal((n_obs, n_models)) * 0.01
        res = superior_predictive_ability(r, n_samples=300, block_length=1.0, seed=s)
        p["rc"].append(res.reality_check_pvalue)
        p["consistent"].append(res.pvalue_consistent)
        p["upper"].append(res.pvalue_upper)
    return {k: float(np.mean(np.array(v) <= 0.10)) for k, v in p.items()}


def test_pvalues_are_approximately_uniform_for_pure_noise() -> None:
    """All strategies at the null boundary: rejection at 10% should be ~10%."""
    rates = _null_rejection_rates(n_sims=150, n_obs=400, n_models=10, seed=7)
    for name, rate in rates.items():
        assert 0.03 <= rate <= 0.19, (name, rate)  # 150 sims: +-3 standard errors


@pytest.mark.slow
def test_pvalues_are_uniform_for_pure_noise_large_simulation() -> None:
    rates = _null_rejection_rates(n_sims=1000, n_obs=500, n_models=10, seed=8)
    for name, rate in rates.items():
        assert rate == pytest.approx(0.10, abs=0.03), (name, rate)
