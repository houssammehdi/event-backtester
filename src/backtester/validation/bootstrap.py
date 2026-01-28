"""Stationary bootstrap with automatic block-length selection.

Returns are serially dependent (volatility clusters, positions persist), so resampling
single observations understates the uncertainty of path statistics such as the Sharpe
ratio or the maximum drawdown. The *stationary bootstrap* of Politis & Romano (1994)
resamples blocks of consecutive observations whose lengths are geometric with mean
``b``; unlike fixed-length block bootstraps the resampled series is itself stationary.
Blocks wrap around the end of the sample.

The mean block length is chosen with the automatic rule of Politis & White (2004), as
corrected by Patton, Politis & White (2009), which minimises the asymptotic mean
squared error of the bootstrap estimate of the long-run variance:

``b_SB = (2 G^2 / D_SB)^(1/3) n^(1/3)``, ``G = sum_k lambda(k/M) |k| R(k)``,
``D_SB = 2 g(0)^2``, ``g(0) = sum_k lambda(k/M) R(k)``,

with ``R`` the sample autocovariances, ``lambda`` the flat-top lag window and the
bandwidth ``M = 2 m`` picked from the correlogram (Politis, 2003).

References:
    Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the
    American Statistical Association* 89(428), 1303-1313.

    Politis, D. N. and White, H. (2004). Automatic block-length selection for the
    dependent bootstrap. *Econometric Reviews* 23(1), 53-70.

    Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic
    block-length selection for the dependent bootstrap". *Econometric Reviews* 28(4),
    372-375.

    Politis, D. N. (2003). Adaptive bandwidth choice. *Journal of Nonparametric
    Statistics* 15(4-5), 517-533.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Statistic = Callable[[FloatArray], FloatArray]
"""Maps a ``(n_samples, n_obs)`` array of return paths to one value per path."""
Seed = int | np.random.Generator | None
IntervalMethod = Literal["percentile", "basic"]

_CHUNK = 256
"""Resamples generated per batch; bounds memory at ``_CHUNK * n_obs`` integers."""


# ---------------------------------------------------------------------- block length
@dataclass(frozen=True, slots=True)
class BlockLength:
    """Estimated optimal mean block lengths (in observations)."""

    stationary: float
    """For the stationary bootstrap (expected block length)."""
    circular: float
    """For the circular block bootstrap (round up to an integer to use it)."""
    bandwidth: int
    """The lag-window bandwidth ``M`` chosen from the correlogram."""


def _flat_top(t: FloatArray) -> FloatArray:
    """Politis' trapezoidal flat-top lag window."""
    a = np.abs(t)
    out: FloatArray = np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))
    return out


def _autocovariances(x: FloatArray, max_lag: int) -> FloatArray:
    """``R(k) = n^-1 sum_t (x_t - mean)(x_{t+k} - mean)`` for ``k = 0..max_lag``."""
    e = x - x.mean()
    n = len(e)
    return np.array([float(e[: n - k] @ e[k:]) / n for k in range(max_lag + 1)])


def optimal_block_length(x: npt.ArrayLike) -> BlockLength:
    """Politis-White (2004) block length with the Patton-Politis-White (2009) correction.

    Tuning follows the authors' reference implementation: ``c = 2``,
    ``K_N = max(5, ceil(sqrt(log10 n)))``, ``m_max = ceil(sqrt n) + K_N`` and a cap of
    ``ceil(min(3 sqrt n, n / 3))``. ``m`` is the smallest lag from which ``K_N``
    consecutive sample autocorrelations are insignificant, and ``M = min(2 m, m_max)``.
    The result is floored at one observation (no dependence: resample single returns).

    Args:
        x: One series; NaNs are dropped.

    Raises:
        ValueError: for fewer than 12 observations.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 12:
        raise ValueError("optimal_block_length needs at least 12 observations")
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    acov = _autocovariances(arr, m_max)
    if acov[0] <= 0:
        return BlockLength(1.0, 1.0, 0)
    critical = 2.0 * math.sqrt(math.log10(n) / n)
    insignificant = np.abs(acov[1:] / acov[0]) < critical  # lags 1..m_max
    m_hat: int | None = None
    for m in range(1, m_max - k_n + 2):
        if insignificant[m - 1 : m - 1 + k_n].all():
            m_hat = m
            break
    bandwidth = m_max if m_hat is None else min(2 * m_hat, m_max)
    lags = np.arange(-bandwidth, bandwidth + 1)
    weights = _flat_top(lags / bandwidth)
    r = acov[np.abs(lags)]
    g0 = float(np.sum(weights * r))
    big_g = float(np.sum(weights * np.abs(lags) * r))
    if g0 == 0:
        return BlockLength(float(b_max), float(b_max), bandwidth)
    cube_root_n = n ** (1.0 / 3.0)
    b_sb = (2.0 * big_g**2 / (2.0 * g0**2)) ** (1.0 / 3.0) * cube_root_n
    b_cb = (2.0 * big_g**2 / (4.0 / 3.0 * g0**2)) ** (1.0 / 3.0) * cube_root_n
    return BlockLength(
        stationary=float(min(max(b_sb, 1.0), b_max)),
        circular=float(min(max(b_cb, 1.0), b_max)),
        bandwidth=bandwidth,
    )


# ---------------------------------------------------------------------- resampling
def _rng(seed: Seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def stationary_bootstrap_indices(
    n_obs: int,
    block_length: float,
    n_samples: int,
    seed: Seed = None,
) -> IntArray:
    """Index paths of the stationary bootstrap, one resample per row.

    Each path starts at a uniformly random observation; every later step starts a new
    block at a random observation with probability ``1 / block_length`` and otherwise
    moves to the next observation, wrapping around the end of the sample.

    Returns:
        An ``(n_samples, n_obs)`` integer array of positions into the original series.
    """
    if n_obs < 1 or n_samples < 1:
        raise ValueError("n_obs and n_samples must be positive")
    if not block_length >= 1:
        raise ValueError("block_length must be at least 1")
    rng = _rng(seed)
    p = 1.0 / block_length
    starts = rng.integers(0, n_obs, size=(n_samples, n_obs))
    new_block = rng.random((n_samples, n_obs)) < p
    new_block[:, 0] = True
    steps = np.arange(n_obs)
    last = np.where(new_block, steps, 0)
    np.maximum.accumulate(last, axis=1, out=last)
    rows = np.arange(n_samples)[:, None]
    out: IntArray = (starts[rows, last] + (steps - last)) % n_obs
    return out


def resample_counts(indices: IntArray, n_obs: int) -> FloatArray:
    """How often each observation appears in each resample (rows of ``indices``).

    Means of resampled series are then one matrix product: ``counts @ x / n_obs``.
    """
    n_samples = indices.shape[0]
    flat = (np.arange(n_samples)[:, None] * n_obs + indices).ravel()
    counts = np.bincount(flat, minlength=n_samples * n_obs)
    return counts.reshape(n_samples, n_obs).astype(np.float64)


# ---------------------------------------------------------------------- statistics
def _annual(periods_per_year: float) -> float:
    return math.sqrt(periods_per_year)


def _equity_paths(paths: FloatArray) -> FloatArray:
    growth = np.cumprod(1.0 + paths, axis=1)
    ones = np.ones((paths.shape[0], 1))
    out: FloatArray = np.hstack([ones, growth])
    return out


def _mean(paths: FloatArray) -> FloatArray:
    out: FloatArray = paths.mean(axis=1)
    return out


def _volatility(periods_per_year: float) -> Statistic:
    def stat(paths: FloatArray) -> FloatArray:
        out: FloatArray = paths.std(axis=1, ddof=1) * _annual(periods_per_year)
        return out

    return stat


def _sharpe(periods_per_year: float) -> Statistic:
    def stat(paths: FloatArray) -> FloatArray:
        sd = paths.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out: FloatArray = paths.mean(axis=1) / sd * _annual(periods_per_year)
        return out

    return stat


def _sortino(periods_per_year: float) -> Statistic:
    def stat(paths: FloatArray) -> FloatArray:
        downside = np.sqrt(np.mean(np.minimum(paths, 0.0) ** 2, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            out: FloatArray = paths.mean(axis=1) / downside * _annual(periods_per_year)
        return out

    return stat


def _total_return(paths: FloatArray) -> FloatArray:
    out: FloatArray = np.prod(1.0 + paths, axis=1) - 1.0
    return out


def _cagr(periods_per_year: float) -> Statistic:
    def stat(paths: FloatArray) -> FloatArray:
        growth = np.prod(1.0 + paths, axis=1)
        years = paths.shape[1] / periods_per_year
        with np.errstate(invalid="ignore"):
            out: FloatArray = np.where(growth > 0, np.abs(growth) ** (1.0 / years) - 1.0, -1.0)
        return out

    return stat


def _max_drawdown(paths: FloatArray) -> FloatArray:
    equity = _equity_paths(paths)
    peak = np.maximum.accumulate(equity, axis=1)
    out: FloatArray = np.max(1.0 - equity / peak, axis=1)
    return out


def default_statistics(periods_per_year: float = 252.0) -> dict[str, Statistic]:
    """The statistics bootstrapped by default, with the package's metric definitions.

    ``sharpe``, ``sortino`` and ``volatility`` are annualised; ``cagr`` uses
    ``years = n / periods_per_year``; ``max_drawdown`` is measured on the compounded
    path starting from 1 (so a loss on the first observation counts).
    """
    return {
        "sharpe": _sharpe(periods_per_year),
        "cagr": _cagr(periods_per_year),
        "max_drawdown": _max_drawdown,
        "volatility": _volatility(periods_per_year),
        "sortino": _sortino(periods_per_year),
        "total_return": _total_return,
        "mean": _mean,
    }


@dataclass(frozen=True, slots=True)
class Interval:
    """A bootstrap confidence interval for one statistic."""

    estimate: float
    lower: float
    upper: float
    std_error: float

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval."""
        return self.lower <= value <= self.upper


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Output of :func:`bootstrap_statistics`."""

    intervals: dict[str, Interval]
    samples: dict[str, FloatArray] = field(repr=False)
    """The bootstrap distribution of every statistic (one value per resample)."""
    block_length: float
    n_samples: int
    confidence: float
    method: IntervalMethod
    n_obs: int

    def __getitem__(self, name: str) -> Interval:
        return self.intervals[name]

    def table(self) -> pd.DataFrame:
        """One row per statistic: estimate, bounds and standard error."""
        rows = {
            name: {
                "estimate": iv.estimate,
                "lower": iv.lower,
                "upper": iv.upper,
                "std_error": iv.std_error,
            }
            for name, iv in self.intervals.items()
        }
        return pd.DataFrame.from_dict(rows, orient="index")


def bootstrap_statistics(
    returns: npt.ArrayLike,
    statistics: Mapping[str, Statistic] | None = None,
    *,
    n_samples: int = 2000,
    block_length: float | None = None,
    confidence: float = 0.95,
    method: IntervalMethod = "percentile",
    periods_per_year: float = 252.0,
    seed: Seed = 0,
) -> BootstrapResult:
    """Stationary-bootstrap confidence intervals for statistics of a return series.

    Args:
        returns: Per-period simple returns; NaNs are dropped.
        statistics: ``name -> function`` of an ``(n_samples, n_obs)`` array of
            resampled paths (defaults to :func:`default_statistics`).
        n_samples: Number of bootstrap resamples.
        block_length: Mean block length; ``None`` selects it with
            :func:`optimal_block_length`.
        confidence: Two-sided coverage of the intervals.
        method: ``"percentile"`` uses the quantiles of the bootstrap distribution;
            ``"basic"`` reflects them around the estimate (``2 theta - q``).
        periods_per_year: Annualisation of the default statistics.
        seed: Random seed or generator; the same seed gives the same intervals.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 12:
        raise ValueError("bootstrap_statistics needs at least 12 observations")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if method not in ("percentile", "basic"):
        raise ValueError(f"unknown interval method {method!r}")
    stats = dict(statistics) if statistics is not None else default_statistics(periods_per_year)
    b = optimal_block_length(r).stationary if block_length is None else float(block_length)
    rng = _rng(seed)
    draws: dict[str, list[FloatArray]] = {name: [] for name in stats}
    done = 0
    while done < n_samples:
        size = min(_CHUNK, n_samples - done)
        paths = r[stationary_bootstrap_indices(n, b, size, rng)]
        for name, fn in stats.items():
            draws[name].append(np.asarray(fn(paths), dtype=np.float64))
        done += size
    alpha = 1.0 - confidence
    original = r[None, :]
    intervals: dict[str, Interval] = {}
    samples: dict[str, FloatArray] = {}
    for name, fn in stats.items():
        values = np.concatenate(draws[name])
        samples[name] = values
        estimate = float(np.asarray(fn(original), dtype=np.float64)[0])
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            intervals[name] = Interval(estimate, math.nan, math.nan, math.nan)
            continue
        lo_q, hi_q = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        if method == "basic":
            lo_q, hi_q = 2.0 * estimate - hi_q, 2.0 * estimate - lo_q
        intervals[name] = Interval(
            estimate=estimate,
            lower=float(lo_q),
            upper=float(hi_q),
            std_error=float(finite.std(ddof=1)) if len(finite) > 1 else math.nan,
        )
    return BootstrapResult(
        intervals=intervals,
        samples=samples,
        block_length=b,
        n_samples=n_samples,
        confidence=confidence,
        method=method,
        n_obs=n,
    )
