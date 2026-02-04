"""Quadratic programming on the capped simplex, by a primal active-set method.

Long-only minimum-variance and mean-variance portfolios solve

``minimise 1/2 x'Qx - c'x  subject to  sum(x) = 1,  lower <= x <= upper``

with ``Q`` positive definite. :func:`solve_simplex_qp` uses the primal active-set
method (Nocedal & Wright, 2006, Algorithm 16.3): it keeps a *working set* of bounds
held active, solves the equality-constrained subproblem on the free variables through
its KKT system, steps as far as feasibility allows (adding the blocking bound), and
when the step vanishes releases the bound whose Lagrange multiplier has the wrong
sign. With ``Q`` positive definite the objective decreases strictly between working
sets, so the method stops at the exact optimum after finitely many iterations - no
step sizes or tolerances on the objective. Optimality is certified by the KKT
conditions, which the returned :class:`QPSolution` reports.

Reference:
    Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.
    Springer. Section 16.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_TOL = 1e-12


@dataclass(frozen=True, slots=True)
class QPSolution:
    """Output of :func:`solve_simplex_qp`."""

    x: FloatArray
    objective: float
    iterations: int
    multiplier: float
    """Lagrange multiplier ``nu`` of the budget constraint (``g_i = nu`` on free
    variables, where ``g = Qx - c``)."""
    kkt_residual: float
    """Largest violation of the KKT conditions at ``x`` (0 at the exact optimum)."""


def _kkt_residual(
    x: FloatArray, g: FloatArray, nu: float, lower: FloatArray, upper: FloatArray
) -> float:
    at_lower = x <= lower + 1e-10
    at_upper = x >= upper - 1e-10
    free = ~(at_lower | at_upper)
    stationarity = np.abs(g[free] - nu).max(initial=0.0)
    lower_sign = np.maximum(nu - g[at_lower & ~at_upper], 0.0).max(initial=0.0)
    upper_sign = np.maximum(g[at_upper & ~at_lower] - nu, 0.0).max(initial=0.0)
    return float(max(stationarity, lower_sign, upper_sign, abs(x.sum() - 1.0)))


def _initial_point(lower: FloatArray, upper: FloatArray) -> FloatArray:
    """A feasible point: fill capacity above the lower bounds proportionally."""
    room = upper - lower
    need = 1.0 - lower.sum()
    total = room.sum()
    return lower + room * (need / total) if total > 0 else lower.copy()


def solve_simplex_qp(
    q: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    *,
    lower: npt.ArrayLike | float = 0.0,
    upper: npt.ArrayLike | float = 1.0,
    max_iter: int | None = None,
) -> QPSolution:
    """Minimise ``1/2 x'Qx - c'x`` s.t. ``sum(x) = 1`` and ``lower <= x <= upper``.

    Args:
        q: Symmetric positive definite ``(n, n)`` matrix.
        c: Linear term (default zero: minimum variance).
        lower: Lower bounds (scalar or per variable).
        upper: Upper bounds (scalar or per variable).
        max_iter: Iteration cap (default ``50 n + 100``); reaching it raises.

    Raises:
        ValueError: for inconsistent bounds, an infeasible budget, or a matrix that is
            not positive definite on the free variables.
        RuntimeError: if the iteration cap is reached (it should not be for a
            positive definite ``Q``).
    """
    qm = np.asarray(q, dtype=np.float64)
    n = qm.shape[0]
    if qm.shape != (n, n) or n == 0:
        raise ValueError("q must be a non-empty square matrix")
    if not np.allclose(qm, qm.T, rtol=1e-10, atol=1e-14):
        raise ValueError("q must be symmetric")
    cv = np.zeros(n) if c is None else np.asarray(c, dtype=np.float64).ravel()
    lo = np.broadcast_to(np.asarray(lower, dtype=np.float64), (n,)).copy()
    hi = np.broadcast_to(np.asarray(upper, dtype=np.float64), (n,)).copy()
    if cv.shape != (n,) or np.any(lo > hi) or not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError("c, lower and upper must have n finite entries with lower <= upper")
    if lo.sum() > 1.0 + 1e-12 or hi.sum() < 1.0 - 1e-12:
        raise ValueError("infeasible: the bounds do not admit weights summing to 1")
    x = _initial_point(lo, hi)
    # working set: -1 = held at lower bound, +1 = held at upper bound, 0 = free
    work = np.zeros(n, dtype=np.int8)
    work[np.isclose(x, lo, rtol=0.0, atol=_TOL)] = -1
    work[np.isclose(x, hi, rtol=0.0, atol=_TOL) & (work == 0)] = 1
    cap = max_iter if max_iter is not None else 50 * n + 100
    scale = max(float(np.abs(qm).max()), float(np.abs(cv).max()), 1e-300)
    nu = 0.0
    for iteration in range(1, cap + 1):
        g = qm @ x - cv
        free = np.flatnonzero(work == 0)
        if len(free):
            k = len(free)
            kkt = np.zeros((k + 1, k + 1))
            kkt[:k, :k] = qm[np.ix_(free, free)]
            kkt[:k, k] = kkt[k, :k] = 1.0
            rhs = np.zeros(k + 1)
            rhs[:k] = -g[free]
            try:
                sol = np.linalg.solve(kkt, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError("q is not positive definite on the free variables") from exc
            p = np.zeros(n)
            p[free] = sol[:k]
            nu = float(-sol[k])  # multiplier of sum(x) = 1 at x + p
        else:
            p = np.zeros(n)
        if np.abs(p).max(initial=0.0) <= 1e-13 * max(1.0, float(np.abs(x).max())):
            g = qm @ x - cv
            if len(free) == 0:
                # no free variable pins nu: choose the largest value that keeps every
                # lower-bound multiplier non-negative, then test the upper bounds
                held_low = g[work == -1]
                nu = float(held_low.min()) if len(held_low) else float(g[work == 1].max())
            mult = np.where(work == -1, g - nu, np.where(work == 1, nu - g, 0.0))
            worst = int(np.argmin(mult))
            if mult[worst] >= -1e-12 * scale:
                return QPSolution(
                    x=x,
                    objective=float(0.5 * x @ qm @ x - cv @ x),
                    iterations=iteration,
                    multiplier=nu,
                    kkt_residual=_kkt_residual(x, g, nu, lo, hi),
                )
            work[worst] = 0
            continue
        alpha = 1.0
        blocking = -1
        for i in free:
            if p[i] < 0:
                step = (lo[i] - x[i]) / p[i]
            elif p[i] > 0:
                step = (hi[i] - x[i]) / p[i]
            else:
                continue
            if step < alpha:
                alpha, blocking = max(step, 0.0), int(i)
        x = x + alpha * p
        if blocking >= 0:
            if p[blocking] < 0:
                x[blocking] = lo[blocking]
                work[blocking] = -1
            else:
                x[blocking] = hi[blocking]
                work[blocking] = 1
    raise RuntimeError(f"active-set QP did not converge in {cap} iterations")
