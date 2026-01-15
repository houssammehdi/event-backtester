"""Multiple-testing aware Sharpe ratio statistics.

Selecting the best of many backtests inflates its Sharpe ratio: with enough trials a
strategy with no skill will look good by luck. The *deflated Sharpe ratio* (Bailey &
Lopez de Prado, 2014) is the probability that the true Sharpe ratio exceeds the Sharpe
ratio one would expect from the best of ``N`` skill-less trials, correcting for sample
length and for non-normal (skewed, fat-tailed) returns.

All Sharpe ratios in this module are **per period** (not annualised).
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sample_moments(returns: npt.ArrayLike) -> tuple[float, float, float, int]:
    """Per-period Sharpe ratio, skewness and (non-excess) kurtosis of ``returns``."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3:
        return math.nan, math.nan, math.nan, n
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd == 0:
        return math.nan, math.nan, math.nan, n
    centred = r - mean
    m2 = float(np.mean(centred**2))
    skew = float(np.mean(centred**3)) / m2**1.5
    kurt = float(np.mean(centred**4)) / m2**2
    return mean / sd, skew, kurt, n


def probabilistic_sharpe_ratio(
    sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_sharpe``.

    ``PSR = Phi((SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + (kurtosis - 1) / 4 * SR^2))``
    """
    if n_obs < 2 or any(math.isnan(x) for x in (sharpe, benchmark_sharpe, skew, kurtosis)):
        return math.nan
    denom = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return math.nan
    z = (sharpe - benchmark_sharpe) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` skill-less strategies.

    ``E[max SR] ~ sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))`` with ``g`` the
    Euler-Mascheroni constant and ``Z`` the standard normal quantile function.
    """
    if n_trials <= 1 or sharpe_variance <= 0 or math.isnan(sharpe_variance):
        return 0.0
    a = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    b = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(
    returns: npt.ArrayLike,
    trial_sharpes: npt.ArrayLike,
) -> float:
    """Deflated Sharpe ratio of the selected strategy's ``returns``.

    Args:
        returns: Per-period returns of the strategy that was selected.
        trial_sharpes: Per-period Sharpe ratios of *every* configuration tried
            (including the selected one). Their dispersion sets the bar.

    Returns:
        A probability in ``[0, 1]``; values above ~0.95 suggest the result is unlikely
        to be a pure artefact of the search.
    """
    sharpe, skew, kurt, n = sample_moments(returns)
    trials = np.asarray(trial_sharpes, dtype=np.float64)
    trials = trials[np.isfinite(trials)]
    variance = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    threshold = expected_max_sharpe(len(trials), variance)
    return probabilistic_sharpe_ratio(sharpe, threshold, n, skew, kurt)
