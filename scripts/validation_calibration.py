"""Calibration of the validation statistics on simulated data with a known answer.

* Size of White's Reality Check and Hansen's SPA: every strategy is pure noise, so a
  test at level a should reject in about a of the simulations.
* Power of the SPA as poor strategies are added (Hansen's motivation for studentising
  and recentring): one skilled strategy plus k clearly bad ones.
* Coverage of stationary-bootstrap percentile intervals on iid normal returns.
* The PBO of pure noise (1/2 in expectation) and of one skilled strategy among noise,
  for two skill levels and two sample lengths.

Every simulation is seeded. The full run takes a few minutes on a laptop-class CPU.

Run:  python scripts/validation_calibration.py
"""

from __future__ import annotations

import math

import numpy as np

from backtester.analytics.report import table
from backtester.validation import (
    bootstrap_statistics,
    default_statistics,
    probability_of_backtest_overfitting,
    superior_predictive_ability,
)

LEVELS = (0.05, 0.10)


def spa_size(n_sims: int, n_obs: int, n_models: int, seed: int) -> dict[str, list[float]]:
    """Rejection rates at LEVELS when every strategy has zero mean differential."""
    rng = np.random.default_rng(seed)
    p: dict[str, list[float]] = {"RC": [], "SPA lower": [], "SPA consistent": [], "SPA upper": []}
    for s in range(n_sims):
        r = rng.standard_normal((n_obs, n_models)) * 0.01
        res = superior_predictive_ability(r, n_samples=500, block_length=1.0, seed=s)
        p["RC"].append(res.reality_check_pvalue)
        p["SPA lower"].append(res.pvalue_lower)
        p["SPA consistent"].append(res.pvalue_consistent)
        p["SPA upper"].append(res.pvalue_upper)
    return {k: [float(np.mean(np.array(v) <= a)) for a in LEVELS] for k, v in p.items()}


def spa_power(n_bad: int, n_sims: int, seed: int) -> dict[str, float]:
    """Rejection rate at 5% with one skilled strategy and ``n_bad`` clearly worse ones."""
    rng = np.random.default_rng(seed)
    rejections = {"RC": 0, "SPA consistent": 0}
    for s in range(n_sims):
        r = rng.standard_normal((500, 1 + n_bad)) * 0.01
        r[:, 0] += 0.0012  # per-period Sharpe 0.12
        r[:, 1:] -= 0.004  # the others lose 0.4% a period
        res = superior_predictive_ability(r, n_samples=500, block_length=1.0, seed=s)
        rejections["RC"] += res.reality_check_pvalue <= 0.05
        rejections["SPA consistent"] += res.pvalue_consistent <= 0.05
    return {k: v / n_sims for k, v in rejections.items()}


def coverage(confidence: float, reps: int, seed: int) -> dict[str, float]:
    """Share of intervals containing the true mean, Sharpe ratio and volatility."""
    rng = np.random.default_rng(seed)
    mu, sd, n = 0.0005, 0.01, 500
    truth = {"mean": mu, "sharpe": mu / sd * math.sqrt(252), "volatility": sd * math.sqrt(252)}
    stats = {k: v for k, v in default_statistics().items() if k in truth}
    hits = dict.fromkeys(truth, 0)
    for s in range(reps):
        sample = rng.normal(mu, sd, n)
        res = bootstrap_statistics(sample, stats, n_samples=500, confidence=confidence, seed=s)
        for k, value in truth.items():
            hits[k] += res[k].contains(value)
    return {k: v / reps for k, v in hits.items()}


def pbo_noise(reps: int, seed: int) -> float:
    """Mean PBO over ``reps`` families of 20 pure-noise strategies."""
    rng = np.random.default_rng(seed)
    values = [
        probability_of_backtest_overfitting(rng.standard_normal((1000, 20)), n_splits=10).pbo
        for _ in range(reps)
    ]
    return float(np.mean(values))


def pbo_skill(sharpe: float, n_obs: int, seeds: int) -> tuple[float, float]:
    """(mean, max) PBO over seeds: one strategy with per-period Sharpe ``sharpe`` among 19."""
    values = []
    for seed in range(seeds):
        r = np.random.default_rng(seed).standard_normal((n_obs, 20))
        r[:, 3] += sharpe
        values.append(probability_of_backtest_overfitting(r, n_splits=10).pbo)
    return float(np.mean(values)), float(np.max(values))


def main() -> None:
    """Run every study and print its table."""
    rows = []
    for m, sims in ((10, 1000), (1, 4000)):
        for name, rates in spa_size(sims, 500, m, seed=m).items():
            family = f"{m} strategies" if m > 1 else "1 strategy"
            rows.append((f"{name}, {family}", *(f"{x:.1%}" for x in rates)))
    print("Size of the data-snooping tests: rejection rate under the null")
    print("(500 observations, iid, 500 bootstrap resamples; 1000 and 4000 simulations)")
    print(table(rows, header=("test", "at 5%", "at 10%")))
    print()
    rows = []
    for n_bad in (0, 5, 20, 50):
        power = spa_power(n_bad, 300, seed=100 + n_bad)
        rows.append((str(n_bad), f"{power['RC']:.0%}", f"{power['SPA consistent']:.0%}"))
    print("Power at 5%: one strategy with per-period Sharpe 0.12, plus k losing strategies")
    print("(500 observations, 300 simulations each)")
    print(table(rows, header=("k", "Reality Check", "SPA consistent")))
    print()
    rows = []
    for level in (0.90, 0.95):
        cov = coverage(level, 1000, seed=int(level * 100))
        rows.append((f"{level:.0%}", *(f"{cov[k]:.1%}" for k in ("mean", "sharpe", "volatility"))))
    print("Coverage of stationary-bootstrap percentile intervals, iid normal returns")
    print("(500 observations, 500 resamples, 1000 samples)")
    print(table(rows, header=("nominal", "mean", "Sharpe", "volatility")))
    print()
    print(f"PBO of 20 pure-noise strategies, mean over 200 families: {pbo_noise(200, 5):.3f}")
    rows = []
    for sharpe in (0.15, 0.2):
        for n_obs in (1000, 2000):
            mean, worst = pbo_skill(sharpe, n_obs, 40)
            rows.append((f"{sharpe:g}", str(n_obs), f"{mean:.3f}", f"{worst:.3f}"))
    print("PBO with one skilled strategy among 19 noise strategies (40 seeds, 10 blocks)")
    print(table(rows, header=("per-period Sharpe", "observations", "mean PBO", "max PBO")))


if __name__ == "__main__":
    main()
