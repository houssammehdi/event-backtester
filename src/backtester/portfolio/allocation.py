"""Portfolio construction: from a covariance matrix to long-only weights.

Every method maps a covariance matrix (and, for mean-variance, expected returns) to
long-only weights that sum to one:

* :func:`inverse_volatility_weights` - ``w_i ~ 1 / sigma_i``.
* :func:`minimum_variance_weights` / :func:`mean_variance_weights` - quadratic
  programs solved exactly by the active-set method of
  :mod:`backtester.portfolio.optimize`, with an optional cap per asset.
* :func:`equal_risk_contribution_weights` - every asset contributes the same share
  ``b_i`` of portfolio variance, ``w_i (Sigma w)_i / w'Sigma w = b_i``. Found through
  the strictly convex problem ``min 1/2 y'Sigma y - sum b_i log y_i`` of Spinu (2013),
  whose minimiser rescaled to ``sum(y) = 1`` has exactly these risk contributions,
  solved by Newton's method with a positivity-preserving backtracking line search.
* :func:`hierarchical_risk_parity_weights` - Lopez de Prado (2016): correlation
  distance ``d_ij = sqrt((1 - rho_ij) / 2)``, Euclidean distance between columns of
  ``d``, single-linkage clustering (implemented here from scratch), quasi-
  diagonalisation of the covariance by the dendrogram's leaf order, and recursive
  bisection of that order with inverse-variance allocation inside each half.

:class:`Allocator` wraps a method with a covariance estimator so that a strategy can
call it on a window of returns.

References:
    Lopez de Prado, M. (2016). Building diversified portfolios that outperform out of
    sample. *Journal of Portfolio Management* 42(4), 59-69.

    Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.

    Maillard, S., Roncalli, T. and Teiletche, J. (2010). The properties of equally
    weighted risk contribution portfolios. *Journal of Portfolio Management* 36(4),
    60-70.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from backtester.portfolio.covariance import LedoitWolf
from backtester.portfolio.optimize import solve_simplex_qp

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
CovarianceEstimator = Callable[[FloatArray], FloatArray]


def _covariance(cov: npt.ArrayLike) -> FloatArray:
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(c).all() or np.any(np.diag(c) <= 0):
        raise ValueError("covariance must be finite with positive variances")
    return c


def _cap(n: int, max_weight: float | None) -> float:
    if max_weight is None:
        return 1.0
    if not 0 < max_weight <= 1 or max_weight * n < 1 - 1e-12:
        raise ValueError("max_weight must be in (0, 1] and at least 1/n")
    return max_weight


def risk_contributions(weights: npt.ArrayLike, cov: npt.ArrayLike) -> FloatArray:
    """Share of portfolio variance contributed by each asset: ``w_i (Sigma w)_i / w'Sigma w``."""
    w = np.asarray(weights, dtype=np.float64)
    c = _covariance(cov)
    marginal = c @ w
    out: FloatArray = w * marginal / float(w @ marginal)
    return out


def inverse_volatility_weights(cov: npt.ArrayLike) -> FloatArray:
    """Weights proportional to ``1 / sigma_i``."""
    inv = 1.0 / np.sqrt(np.diag(_covariance(cov)))
    out: FloatArray = inv / inv.sum()
    return out


def inverse_variance_weights(cov: npt.ArrayLike) -> FloatArray:
    """Weights proportional to ``1 / sigma_i^2``.

    This is the minimum-variance portfolio when the assets are uncorrelated.
    """
    inv = 1.0 / np.diag(_covariance(cov))
    out: FloatArray = inv / inv.sum()
    return out


def minimum_variance_weights(cov: npt.ArrayLike, *, max_weight: float | None = None) -> FloatArray:
    """Long-only minimum-variance weights, optionally capped at ``max_weight``."""
    c = _covariance(cov)
    return solve_simplex_qp(c, upper=_cap(len(c), max_weight)).x


def mean_variance_weights(
    expected_returns: npt.ArrayLike,
    cov: npt.ArrayLike,
    *,
    risk_aversion: float = 1.0,
    max_weight: float | None = None,
) -> FloatArray:
    """Long-only weights maximising ``mu'w - risk_aversion / 2 * w'Sigma w``."""
    c = _covariance(cov)
    mu = np.asarray(expected_returns, dtype=np.float64).ravel()
    if mu.shape != (len(c),) or not np.isfinite(mu).all():
        raise ValueError("expected_returns must have one finite value per asset")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    return solve_simplex_qp(risk_aversion * c, mu, upper=_cap(len(c), max_weight)).x


def equal_risk_contribution_weights(
    cov: npt.ArrayLike,
    budgets: npt.ArrayLike | None = None,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> FloatArray:
    """Risk-parity weights: asset ``i`` contributes ``budgets[i]`` of the variance.

    Solves Spinu's convex problem ``min_y 1/2 y'Sigma y - sum_i b_i log y_i`` with
    Newton's method (gradient ``Sigma y - b / y``, Hessian ``Sigma + diag(b / y^2)``);
    the step is halved until it keeps ``y > 0`` and satisfies the Armijo condition.
    At the minimiser ``y_i (Sigma y)_i = b_i``, so ``w = y / sum(y)`` has risk
    contributions ``b_i / sum(b)``.

    Args:
        cov: Covariance matrix.
        budgets: Positive risk budgets (default equal: the ERC portfolio).
        tol: Convergence tolerance on the relative risk-contribution error
            ``max_i |y_i (Sigma y)_i - b_i| / b_i``.
        max_iter: Newton iteration cap.

    Raises:
        RuntimeError: if the iterations stall before reaching ``tol`` (it has not
            happened on any positive definite matrix tried; the objective is strictly
            convex and self-concordant up to scaling).
    """
    c = _covariance(cov)
    n = len(c)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=np.float64)
    if b.shape != (n,) or np.any(b <= 0):
        raise ValueError("budgets must be one positive number per asset")
    b = b / b.sum()

    def objective(y: FloatArray) -> float:
        return float(0.5 * y @ c @ y - b @ np.log(y))

    def error(y: FloatArray) -> float:
        return float(np.max(np.abs(y * (c @ y) - b) / b))

    x0 = inverse_volatility_weights(c)
    y = x0 / math.sqrt(float(x0 @ c @ x0))  # scaled so that y'Sigma y = sum(b) = 1
    f = objective(y)
    for _ in range(max_iter):
        if error(y) < tol:
            break
        grad = c @ y - b / y
        step = np.linalg.solve(c + np.diag(b / y**2), grad)
        decrement = float(grad @ step)  # squared Newton decrement: f - f* ~ decrement / 2
        # Once the predicted decrease is below the resolution of f, the Armijo test can
        # no longer be evaluated; Newton is then deep in its quadratic phase and takes
        # full steps (still halved if needed to keep y > 0).
        pure_newton = decrement < 1e-10 * (1.0 + abs(f))
        t = 1.0
        while t >= 1e-12:
            candidate = y - t * step
            if np.all(candidate > 0):
                f_new = objective(candidate)
                if pure_newton or f_new <= f - 1e-4 * t * decrement:
                    break
            t *= 0.5
        else:
            break  # no step decreases f: converged as far as floats allow
        y, f = candidate, f_new
    if error(y) >= max(tol, 1e-8):
        raise RuntimeError("equal risk contribution did not converge")
    out: FloatArray = y / y.sum()
    return out


# ---------------------------------------------------------------------- HRP
def correlation_distance(cov: npt.ArrayLike) -> FloatArray:
    """``d_ij = sqrt((1 - rho_ij) / 2)``, a proper metric on correlation."""
    c = _covariance(cov)
    sd = np.sqrt(np.diag(c))
    corr = np.clip(c / np.outer(sd, sd), -1.0, 1.0)
    out: FloatArray = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    np.fill_diagonal(out, 0.0)
    return out


def single_linkage(distance: npt.ArrayLike) -> FloatArray:
    """Single-linkage agglomerative clustering of a distance matrix.

    Returns a SciPy-style ``(n - 1, 4)`` linkage matrix: row ``k`` merges clusters
    ``Z[k, 0] < Z[k, 1]`` at distance ``Z[k, 2]`` into cluster ``n + k`` holding
    ``Z[k, 3]`` items. Merges happen in order of increasing distance; ties go to the
    pair of clusters with the smallest ids.
    """
    d = np.asarray(distance, dtype=np.float64)
    n = d.shape[0]
    if d.shape != (n, n) or n < 1:
        raise ValueError("distance must be a square matrix")
    active = list(range(n))  # cluster id held by each slot
    sizes = dict.fromkeys(range(n), 1)
    work = d.copy()
    np.fill_diagonal(work, np.inf)
    alive = np.ones(n, dtype=bool)
    out = np.zeros((max(n - 1, 0), 4))
    for k in range(n - 1):
        masked = np.where(alive[:, None] & alive[None, :], work, np.inf)
        best = float(masked.min())
        candidates = np.argwhere(masked == best)
        pairs = sorted(
            (min(active[i], active[j]), max(active[i], active[j]), int(i), int(j))
            for i, j in candidates
        )
        lo_id, hi_id, i, j = pairs[0]
        keep, drop = (i, j) if active[i] == lo_id else (j, i)
        new_id = n + k
        out[k] = (lo_id, hi_id, best, sizes[lo_id] + sizes[hi_id])
        # single linkage: the distance to the merged cluster is the minimum
        merged = np.minimum(work[keep], work[drop])
        work[keep, :] = merged
        work[:, keep] = merged
        work[keep, keep] = np.inf
        alive[drop] = False
        sizes[new_id] = sizes.pop(lo_id) + sizes.pop(hi_id)
        active[keep] = new_id
    return out


def quasi_diagonal_order(linkage: npt.ArrayLike) -> list[int]:
    """Leaf order of a dendrogram: each cluster lists its first child, then its second."""
    z = np.asarray(linkage, dtype=np.float64)
    n = len(z) + 1
    if len(z) == 0:
        return [0]

    def leaves(cluster: int) -> list[int]:
        order: list[int] = []
        stack = [cluster]
        while stack:
            node = stack.pop()
            if node < n:
                order.append(node)
            else:
                left, right = z[node - n, 0], z[node - n, 1]
                stack.extend((int(right), int(left)))  # pop left first
        return order

    return leaves(2 * n - 2)


def _cluster_variance(cov: FloatArray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def hierarchical_risk_parity_weights(cov: npt.ArrayLike) -> FloatArray:
    """Hierarchical risk parity weights (Lopez de Prado, 2016).

    The recursive bisection splits the quasi-diagonal order into halves (the first
    half gets ``len // 2`` items), weights each half by ``1 - V_half / (V_1 + V_2)``
    with ``V`` the variance of its inverse-variance portfolio, and recurses.

    As published, the weights can depend on the order (labels) of the assets: a
    merge of two single assets lists the smaller label first, and the bisection cuts
    the leaf order by position rather than along the tree, so relabelling the two
    members of a pair that a cut separates changes the weights.
    """
    c = _covariance(cov)
    n = len(c)
    d = correlation_distance(c)
    # the second distance: Euclidean distance between columns of d (computed from the
    # differences directly, which avoids the cancellation of |a|^2 + |b|^2 - 2 a.b)
    euclid = np.sqrt(np.sum((d[:, :, None] - d[:, None, :]) ** 2, axis=0))
    order = quasi_diagonal_order(single_linkage(euclid))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [
            part
            for cluster in clusters
            if len(cluster) > 1
            for part in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
        ]
        for first, second in zip(clusters[::2], clusters[1::2], strict=True):
            v1, v2 = _cluster_variance(c, first), _cluster_variance(c, second)
            alpha = 1.0 - v1 / (v1 + v2)
            w[first] *= alpha
            w[second] *= 1.0 - alpha
    return w


# ---------------------------------------------------------------------- allocators
@dataclass(frozen=True, slots=True)
class Allocator(ABC):
    """Maps a window of asset returns to long-only weights that sum to one.

    Attributes:
        covariance: Covariance estimator applied to the window (default: Ledoit-Wolf
            shrinkage to constant correlation).
    """

    name: ClassVar[str] = "allocator"
    covariance: CovarianceEstimator = field(default_factory=LedoitWolf)

    def __call__(self, returns: npt.ArrayLike) -> FloatArray:
        """Weights for the columns of a ``(T, N)`` window of returns."""
        r = np.asarray(returns, dtype=np.float64)
        if r.ndim != 2 or r.shape[1] == 0:
            raise ValueError("returns must be a (T, N) array with N >= 1")
        if r.shape[1] == 1:
            return np.ones(1)
        return self.weights(r, np.asarray(self.covariance(r), dtype=np.float64))

    @abstractmethod
    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """Weights from the window and its estimated covariance."""


@dataclass(frozen=True, slots=True)
class EqualWeight(Allocator):
    """``1 / N`` in every asset (the covariance is not used)."""

    name: ClassVar[str] = "ew"

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """Equal weights."""
        return np.full(returns.shape[1], 1.0 / returns.shape[1])


@dataclass(frozen=True, slots=True)
class InverseVolatility(Allocator):
    """``w_i ~ 1 / sigma_i``."""

    name: ClassVar[str] = "iv"

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """Inverse-volatility weights."""
        return inverse_volatility_weights(cov)


@dataclass(frozen=True, slots=True)
class MinimumVariance(Allocator):
    """Long-only minimum variance, each weight at most ``max_weight``."""

    name: ClassVar[str] = "minvar"
    max_weight: float | None = None

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """Minimum-variance weights."""
        return minimum_variance_weights(cov, max_weight=self.max_weight)


@dataclass(frozen=True, slots=True)
class MeanVariance(Allocator):
    """Long-only mean-variance with the window's mean returns as expected returns."""

    name: ClassVar[str] = "meanvar"
    risk_aversion: float = 10.0
    max_weight: float | None = None

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """Mean-variance weights."""
        return mean_variance_weights(
            returns.mean(axis=0),
            cov,
            risk_aversion=self.risk_aversion,
            max_weight=self.max_weight,
        )


@dataclass(frozen=True, slots=True)
class EqualRiskContribution(Allocator):
    """Equal risk contribution (risk parity)."""

    name: ClassVar[str] = "erc"

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """ERC weights."""
        return equal_risk_contribution_weights(cov)


@dataclass(frozen=True, slots=True)
class HierarchicalRiskParity(Allocator):
    """Hierarchical risk parity."""

    name: ClassVar[str] = "hrp"

    def weights(self, returns: FloatArray, cov: FloatArray) -> FloatArray:
        """HRP weights."""
        return hierarchical_risk_parity_weights(cov)


ALLOCATORS: dict[str, type[Allocator]] = {
    cls.name: cls
    for cls in (
        EqualWeight,
        InverseVolatility,
        MinimumVariance,
        MeanVariance,
        EqualRiskContribution,
        HierarchicalRiskParity,
    )
}
"""Allocators by short name (as used on the command line)."""
