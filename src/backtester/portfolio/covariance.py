"""Covariance estimation for portfolio construction: sample and Ledoit-Wolf shrinkage.

With ``N`` assets and ``T`` observations the sample covariance has ``N (N + 1) / 2``
free parameters, and when ``T`` is not much larger than ``N`` its extreme eigenvalues
are badly estimated: optimisers then load up on spurious low-variance combinations.
Shrinkage replaces it with the convex combination ``d F + (1 - d) S`` of the sample
covariance ``S`` and a structured target ``F``, with the intensity ``d`` chosen to
minimise the expected Frobenius loss (Ledoit & Wolf, 2003, 2004a, 2004b):

``d* = max(0, min(1, kappa / T))``, ``kappa = (pi - rho) / gamma``

* ``pi`` - sum of the asymptotic variances of the entries of ``S``;
* ``rho`` - sum of the asymptotic covariances between the entries of ``F`` and ``S``;
* ``gamma`` - squared Frobenius distance between ``F`` and ``S`` (misspecification).

Two targets are implemented, each exactly as in the authors' published formulas and
code: the **constant-correlation** target of "Honey, I shrunk the sample covariance
matrix" (sample variances, one average correlation) and the **identity** target of
"A well-conditioned estimator for large-dimensional covariance matrices" (the average
variance times the identity; here ``rho = 0``). As in those papers ``S`` is the maximum
likelihood estimate (divided by ``T``, not ``T - 1``).

References:
    Ledoit, O. and Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix.
    *Journal of Portfolio Management* 30(4), 110-119.

    Ledoit, O. and Wolf, M. (2004b). A well-conditioned estimator for large-dimensional
    covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ShrinkageTarget = Literal["constant_correlation", "identity"]


def _clean(returns: npt.ArrayLike) -> FloatArray:
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("returns must be a 2-D (observations x assets) array")
    if x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("need at least two observations of at least one asset")
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite")
    return x


def sample_covariance(returns: npt.ArrayLike, *, ddof: int = 1) -> FloatArray:
    """Sample covariance of the columns of ``returns`` (unbiased with ``ddof=1``)."""
    x = _clean(returns)
    centred = x - x.mean(axis=0)
    out: FloatArray = centred.T @ centred / (x.shape[0] - ddof)
    return out


@dataclass(frozen=True, slots=True)
class ShrinkageEstimate:
    """Output of :func:`ledoit_wolf`."""

    covariance: FloatArray
    """The shrunk covariance ``d F + (1 - d) S``."""
    shrinkage: float
    """The intensity ``d`` in ``[0, 1]``."""
    target: FloatArray
    """The shrinkage target ``F``."""
    sample: FloatArray
    """The sample covariance ``S`` (divided by ``T``)."""


def ledoit_wolf(
    returns: npt.ArrayLike,
    target: ShrinkageTarget = "constant_correlation",
    *,
    shrinkage: float | None = None,
) -> ShrinkageEstimate:
    """Ledoit-Wolf shrinkage estimate of the covariance of the columns of ``returns``.

    Args:
        returns: ``(T, N)`` array of returns.
        target: ``"constant_correlation"`` (Ledoit & Wolf, 2004a) or ``"identity"``
            (the average variance times the identity; Ledoit & Wolf, 2004b).
        shrinkage: Use this intensity instead of the estimated optimum.
    """
    x = _clean(returns)
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample).copy()
    if target == "constant_correlation":
        sd = np.sqrt(var)
        if n > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = sample / np.outer(sd, sd)
            r_bar = float((np.nansum(corr) - n) / (n * (n - 1)))
        else:
            r_bar = 0.0
        prior = r_bar * np.outer(sd, sd)
        np.fill_diagonal(prior, var)
    elif target == "identity":
        prior = np.eye(n) * float(var.mean())
    else:
        raise ValueError(f"unknown shrinkage target {target!r}")
    if shrinkage is None:
        y = x**2
        pi_mat = y.T @ y / t - sample**2  # asymptotic variances of the entries of S
        pi_hat = float(pi_mat.sum())
        if target == "constant_correlation":
            theta = (x**3).T @ x / t - var[:, None] * sample  # theta_{ii,ij}
            np.fill_diagonal(theta, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(var[:, None] > 0, np.sqrt(var[None, :] / var[:, None]), 0.0)
            rho_hat = float(np.trace(pi_mat)) + r_bar * float(np.sum(ratio * theta))
        else:
            rho_hat = 0.0
        gamma_hat = float(np.sum((sample - prior) ** 2))
        if gamma_hat <= 0:
            intensity = 1.0  # the sample covariance already equals the target
        else:
            intensity = max(0.0, min(1.0, (pi_hat - rho_hat) / gamma_hat / t))
    else:
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        intensity = float(shrinkage)
    covariance = intensity * prior + (1.0 - intensity) * sample
    return ShrinkageEstimate(covariance, intensity, prior, sample)


@dataclass(frozen=True, slots=True)
class SampleCovariance:
    """Covariance estimator: the unbiased sample covariance."""

    def __call__(self, returns: npt.ArrayLike) -> FloatArray:
        """Estimate from a ``(T, N)`` returns window."""
        return sample_covariance(returns)


@dataclass(frozen=True, slots=True)
class LedoitWolf:
    """Covariance estimator: Ledoit-Wolf shrinkage towards ``target``."""

    target: ShrinkageTarget = "constant_correlation"

    def __call__(self, returns: npt.ArrayLike) -> FloatArray:
        """Estimate from a ``(T, N)`` returns window."""
        return ledoit_wolf(returns, self.target).covariance
