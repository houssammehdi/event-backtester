"""Data-snooping tests for a family of strategies: White's Reality Check and Hansen's SPA.

After trying ``m`` strategies on the same data, the best one beats the benchmark by
luck with high probability. Both tests ask whether *any* strategy's expected
performance exceeds the benchmark's, with the null hypothesis

``H0: max_k E[d_k] <= 0``, ``d_k,t = r_k,t - r_benchmark,t``,

and both approximate the distribution of the maximum over the whole family with the
stationary bootstrap (resampling the same time indices for every strategy, which keeps
their cross-correlation).

* **Reality Check** (White, 2000): statistic ``V = max_k sqrt(n) mean(d_k)``, bootstrap
  values ``max_k sqrt(n) (mean*(d_k) - mean(d_k))``. Every strategy is recentred to the
  null boundary, which is conservative when the family contains poor strategies.
* **Superior Predictive Ability** (Hansen, 2005): studentised statistic
  ``T = max(max_k sqrt(n) mean(d_k) / w_k, 0)`` and data-dependent recentring. The
  *consistent* p-value recentres only strategies whose t-statistic is above
  ``-sqrt(2 log log n)``; the *lower* p-value recentres only those with a positive mean
  (liberal); the *upper* p-value recentres all of them (conservative, the least
  favourable configuration). ``p_lower <= p_consistent <= p_upper``.

``w_k^2`` is the stationary-bootstrap variance of ``sqrt(n) mean(d_k)`` in closed
form (Politis & Romano, 1994), as in Hansen (2005):
``w^2 = g_0 + 2 sum_{i=1}^{n-1} k(n, i) g_i`` with
``k(n, i) = (1 - i/n) (1 - q)^i + (i/n) (1 - q)^(n-i)`` and ``q = 1 / block_length``.

p-values are the share of bootstrap statistics at least as large as the sample
statistic. (With continuous data ``>`` and ``>=`` agree, except at the floor: when no
strategy beats the benchmark in sample, ``T = 0`` and most bootstrap values are exactly
0 too, and only ``>=`` gives the correct p-value of 1.)

References:
    White, H. (2000). A reality check for data snooping. *Econometrica* 68(5),
    1097-1126.

    Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business
    & Economic Statistics* 23(4), 365-380.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.validation.bootstrap import (
    Seed,
    _rng,
    optimal_block_length,
    resample_counts,
    stationary_bootstrap_indices,
)

FloatArray = npt.NDArray[np.float64]

_CHUNK = 250


@dataclass(frozen=True, slots=True)
class SPAResult:
    """Output of :func:`superior_predictive_ability`."""

    statistic: float
    """Hansen's ``T_SPA``: the largest studentised mean differential, floored at 0."""
    pvalue_consistent: float
    pvalue_lower: float
    pvalue_upper: float
    reality_check_statistic: float
    """White's ``V``: the largest ``sqrt(n)`` mean differential (not studentised)."""
    reality_check_pvalue: float
    mean_differential: FloatArray = field(repr=False)
    """Per-period mean of ``strategy - benchmark`` returns, per strategy."""
    t_statistics: FloatArray = field(repr=False)
    omega: FloatArray = field(repr=False)
    """Long-run standard deviation of ``sqrt(n)`` times the mean differential."""
    block_length: float
    n_samples: int
    n_obs: int
    labels: tuple[str, ...]

    @property
    def best(self) -> int:
        """Index of the strategy with the largest t-statistic."""
        return int(np.argmax(self.t_statistics))

    @property
    def best_label(self) -> str:
        """Label of :attr:`best`."""
        return self.labels[self.best]


def _differentials(
    returns: npt.ArrayLike | pd.DataFrame, benchmark: npt.ArrayLike | pd.Series | None
) -> tuple[FloatArray, tuple[str, ...]]:
    if isinstance(returns, pd.DataFrame):
        labels = tuple(str(c) for c in returns.columns)
        values = returns.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(returns, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        labels = tuple(str(j) for j in range(values.shape[1]))
    if values.ndim != 2:
        raise ValueError("returns must be a 2-D (periods x strategies) array")
    if benchmark is None:
        bench = np.zeros(values.shape[0])
    else:
        bench = np.asarray(benchmark, dtype=np.float64).ravel()
        if bench.shape[0] != values.shape[0]:
            raise ValueError("benchmark and returns must have the same number of periods")
    d = values - bench[:, None]
    if np.isnan(d).any():
        raise ValueError("returns or benchmark contain NaN")
    return d, labels


def long_run_variance(d: npt.ArrayLike, block_length: float) -> FloatArray:
    """Stationary-bootstrap variance of ``sqrt(n)`` times the column means of ``d``.

    Closed form of Politis & Romano (1994) used by Hansen (2005); autocovariances at
    every lag are computed at once with the FFT.
    """
    x = np.asarray(d, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if not block_length >= 1:
        raise ValueError("block_length must be at least 1")
    e = x - x.mean(axis=0)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(e, n=size, axis=0)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=size, axis=0)[:n] / n
    q = 1.0 / block_length
    i = np.arange(1, n)
    with np.errstate(under="ignore"):
        kappa = (1.0 - i / n) * (1.0 - q) ** i + (i / n) * (1.0 - q) ** (n - i)
    out: FloatArray = acov[0] + 2.0 * (kappa @ acov[1:])
    return np.maximum(out, 0.0)


def superior_predictive_ability(
    returns: npt.ArrayLike | pd.DataFrame,
    benchmark: npt.ArrayLike | pd.Series | None = None,
    *,
    n_samples: int = 1000,
    block_length: float | None = None,
    seed: Seed = 0,
) -> SPAResult:
    """Hansen's SPA test, with White's Reality Check from the same resamples.

    Args:
        returns: ``(T, m)`` per-period returns of the strategy family (a
            :class:`~pandas.DataFrame` keeps the column labels).
        benchmark: Benchmark returns over the same periods; ``None`` means zero
            (cash, or returns already in excess of the benchmark).
        n_samples: Bootstrap resamples.
        block_length: Mean block length of the stationary bootstrap; ``None`` uses
            the median Politis-White estimate over the differential series.
        seed: Random seed or generator.
    """
    d, labels = _differentials(returns, benchmark)
    n, m = d.shape
    if n < 12:
        raise ValueError("need at least 12 periods")
    if block_length is None:
        lengths = [optimal_block_length(d[:, k]).stationary for k in range(m)]
        block_length = float(np.median(lengths))
    d_bar = d.mean(axis=0)
    omega = np.sqrt(long_run_variance(d, block_length))
    root_n = math.sqrt(n)
    # A constant differential has no sampling variance: it is superior with certainty
    # if positive (t = +inf), irrelevant if negative (-inf), identical if zero (0).
    degenerate = omega <= 0
    safe_omega = np.where(degenerate, 1.0, omega)
    certain = np.where(d_bar > 0, np.inf, np.where(d_bar < 0, -np.inf, 0.0))
    t_stats = np.where(degenerate, certain, root_n * d_bar / safe_omega)
    statistic = max(float(np.max(t_stats)), 0.0)
    rc_statistic = float(np.max(root_n * d_bar))

    threshold = -math.sqrt(2.0 * math.log(math.log(n)))
    centres = {
        "lower": np.maximum(d_bar, 0.0),
        "consistent": np.where(t_stats >= threshold, d_bar, 0.0),
        "upper": d_bar,
    }
    exceed = dict.fromkeys(centres, 0)
    rc_exceed = 0
    rng = _rng(seed)
    done = 0
    while done < n_samples:
        size = min(_CHUNK, n_samples - done)
        counts = resample_counts(stationary_bootstrap_indices(n, block_length, size, rng), n)
        d_star = counts @ d / n  # (size, m) resampled means
        rc_values = np.max(root_n * (d_star - d_bar), axis=1)
        rc_exceed += int(np.sum(rc_values >= rc_statistic))
        for name, centre in centres.items():
            z = root_n * (d_star - centre) / safe_omega
            z = np.where(degenerate, np.where(centre == d_bar, 0.0, -np.inf), z)
            values = np.maximum(np.max(z, axis=1), 0.0)
            # >=, not >: both statistics are floored at 0, so ties at 0 are common
            # when every strategy trails the benchmark (T = 0). Counting them keeps
            # the p-value at 1 there instead of spuriously near 0.
            exceed[name] += int(np.sum(values >= statistic))
        done += size
    return SPAResult(
        statistic=statistic,
        pvalue_consistent=exceed["consistent"] / n_samples,
        pvalue_lower=exceed["lower"] / n_samples,
        pvalue_upper=exceed["upper"] / n_samples,
        reality_check_statistic=rc_statistic,
        reality_check_pvalue=rc_exceed / n_samples,
        mean_differential=d_bar,
        t_statistics=t_stats,
        omega=omega,
        block_length=block_length,
        n_samples=n_samples,
        n_obs=n,
        labels=labels,
    )
