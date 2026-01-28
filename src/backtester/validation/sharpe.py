"""Sharpe-ratio inference for non-normal returns.

A Sharpe ratio estimated from a backtest is a random variable. Its sampling error
grows with skewness and fat tails, and selecting the best of many backtests biases
it upwards. The functions here quantify both effects:

* :func:`sharpe_ratio_std_error` - asymptotic standard error of the estimate for iid,
  non-normal returns (Lo, 2002; Mertens, 2002).
* :func:`probabilistic_sharpe_ratio` (PSR) - the probability that the true Sharpe ratio
  exceeds a benchmark (Bailey & Lopez de Prado, 2012).
* :func:`min_track_record_length` (MinTRL) - how many observations are needed before
  the estimate is significantly above the benchmark (Bailey & Lopez de Prado, 2012).
* :func:`deflated_sharpe_ratio` (DSR) - the PSR against the Sharpe ratio one would
  expect from the best of ``N`` skill-less trials (Bailey & Lopez de Prado, 2014).

All Sharpe ratios in this module are **per period** (not annualised); multiply by
``sqrt(periods_per_year)`` to annualise. Kurtosis is the ordinary (non-excess)
kurtosis, 3 for a normal distribution.

References:
    Bailey, D. H. and Lopez de Prado, M. (2012). The Sharpe ratio efficient frontier.
    *Journal of Risk* 15(2), 3-44.

    Bailey, D. H. and Lopez de Prado, M. (2014). The deflated Sharpe ratio: correcting
    for selection bias, backtest overfitting and non-normality. *Journal of Portfolio
    Management* 40(5), 94-107.

    Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*
    58(4), 36-52.

    Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002).
    Working paper, University of Basel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sample_moments(returns: npt.ArrayLike) -> tuple[float, float, float, int]:
    """Per-period Sharpe ratio, skewness and (non-excess) kurtosis of ``returns``.

    The Sharpe ratio uses the sample standard deviation (``ddof=1``); skewness and
    kurtosis use population moments. NaNs are dropped. Returns NaNs (and the count)
    for fewer than three observations or zero variance.
    """
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


def _variance_factor(sharpe: float, skew: float, kurtosis: float) -> float:
    """``1 - skew * SR + (kurtosis - 1) / 4 * SR^2`` (Mertens' correction)."""
    return 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2


def sharpe_ratio_std_error(
    sharpe: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0
) -> float:
    """Asymptotic standard error of an estimated per-period Sharpe ratio.

    ``se = sqrt((1 - skew * SR + (kurtosis - 1) / 4 * SR^2) / (n - 1))``. For normal
    returns this reduces to Lo's (2002) ``sqrt((1 + SR^2 / 2) / (n - 1))``.
    """
    if n_obs < 2 or any(math.isnan(x) for x in (sharpe, skew, kurtosis)):
        return math.nan
    factor = _variance_factor(sharpe, skew, kurtosis)
    if factor <= 0:
        return math.nan
    return math.sqrt(factor / (n_obs - 1))


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
    if math.isnan(benchmark_sharpe):
        return math.nan
    se = sharpe_ratio_std_error(sharpe, n_obs, skew, kurtosis)
    if math.isnan(se):
        return math.nan
    return _NORMAL.cdf((sharpe - benchmark_sharpe) / se)


def min_track_record_length(
    sharpe: float,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float:
    """Minimum track record length, in observations, for ``PSR >= confidence``.

    ``MinTRL = 1 + (1 - skew * SR + (kurtosis - 1) / 4 * SR^2) * (z / (SR - SR*))^2``
    with ``z`` the standard normal quantile of ``confidence``. It is infinite when
    the estimate does not exceed the benchmark: no amount of data at this Sharpe ratio
    would make it significantly better.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if any(math.isnan(x) for x in (sharpe, benchmark_sharpe, skew, kurtosis)):
        return math.nan
    if sharpe <= benchmark_sharpe:
        return math.inf
    factor = _variance_factor(sharpe, skew, kurtosis)
    if factor <= 0:
        return math.nan
    z = _NORMAL.inv_cdf(confidence)
    return 1.0 + factor * (z / (sharpe - benchmark_sharpe)) ** 2


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` skill-less strategies.

    ``E[max SR] ~ sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))`` with ``g`` the
    Euler-Mascheroni constant, ``Z`` the standard normal quantile function and ``V`` the
    variance of the trials' Sharpe ratios (Bailey & Lopez de Prado, 2014).
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
        to be a pure artefact of the search. Correlated trials overstate the effective
        number of independent trials, which makes the DSR conservative.
    """
    sharpe, skew, kurt, n = sample_moments(returns)
    trials = np.asarray(trial_sharpes, dtype=np.float64)
    trials = trials[np.isfinite(trials)]
    variance = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    threshold = expected_max_sharpe(len(trials), variance)
    return probabilistic_sharpe_ratio(sharpe, threshold, n, skew, kurt)


@dataclass(frozen=True, slots=True)
class SharpeInference:
    """Point estimate, uncertainty and significance of one strategy's Sharpe ratio.

    Per-period quantities are suffixed nowhere; annualised ones say so.
    """

    n_obs: int
    periods_per_year: float
    sharpe: float
    """Per-period Sharpe ratio."""
    skew: float
    kurtosis: float
    std_error: float
    """Standard error of the per-period Sharpe ratio (non-normal, iid)."""
    benchmark_sharpe: float
    """Per-period benchmark the PSR and MinTRL are computed against."""
    psr: float
    confidence: float
    min_track_record: float
    """MinTRL in observations at ``confidence``."""
    dsr: float | None = None
    """Deflated Sharpe ratio, when the trials of a search were supplied."""
    n_trials: int | None = None

    @property
    def sharpe_annualized(self) -> float:
        """The Sharpe ratio times ``sqrt(periods_per_year)``."""
        return self.sharpe * math.sqrt(self.periods_per_year)

    @property
    def min_track_record_years(self) -> float:
        """MinTRL expressed in years."""
        return self.min_track_record / self.periods_per_year

    @property
    def track_record_years(self) -> float:
        """Length of the sample in years."""
        return self.n_obs / self.periods_per_year


def sharpe_inference(
    returns: npt.ArrayLike,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 252.0,
    confidence: float = 0.95,
    trial_sharpes: npt.ArrayLike | None = None,
) -> SharpeInference:
    """Collect Sharpe-ratio statistics of ``returns`` in one object.

    Args:
        returns: Per-period returns.
        benchmark_sharpe: Per-period Sharpe ratio to test against (PSR, MinTRL).
        periods_per_year: Used only for the annualised views.
        confidence: Confidence level of the MinTRL.
        trial_sharpes: Per-period Sharpe ratios of every configuration of the search
            that produced ``returns``; enables the deflated Sharpe ratio.
    """
    sharpe, skew, kurt, n = sample_moments(returns)
    dsr: float | None = None
    n_trials: int | None = None
    if trial_sharpes is not None:
        trials = np.asarray(trial_sharpes, dtype=np.float64)
        n_trials = int(np.isfinite(trials).sum())
        dsr = deflated_sharpe_ratio(returns, trials)
    return SharpeInference(
        n_obs=n,
        periods_per_year=periods_per_year,
        sharpe=sharpe,
        skew=skew,
        kurtosis=kurt,
        std_error=sharpe_ratio_std_error(sharpe, n, skew, kurt),
        benchmark_sharpe=benchmark_sharpe,
        psr=probabilistic_sharpe_ratio(sharpe, benchmark_sharpe, n, skew, kurt),
        confidence=confidence,
        min_track_record=min_track_record_length(sharpe, benchmark_sharpe, skew, kurt, confidence),
        dsr=dsr,
        n_trials=n_trials,
    )
