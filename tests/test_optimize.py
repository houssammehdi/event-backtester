"""The active-set QP on the capped simplex."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester.portfolio import solve_simplex_qp


def random_problem(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T / n + np.diag(rng.uniform(0.01, 0.5, n)), rng.normal(0, 0.5, n)


def brute_force(q: np.ndarray, c: np.ndarray, cap: float) -> np.ndarray:
    """Global optimum by enumerating every assignment of variables to {free, 0, cap}."""
    n = len(q)
    best_value, best_x = np.inf, None
    for state in itertools.product((0, 1, 2), repeat=n):
        x = np.array([0.0 if s == 1 else cap if s == 2 else 0.0 for s in state])
        free = [i for i in range(n) if state[i] == 0]
        if free:
            fixed = [i for i in range(n) if state[i] != 0]
            k = len(free)
            kkt = np.zeros((k + 1, k + 1))
            kkt[:k, :k] = q[np.ix_(free, free)]
            kkt[:k, k] = kkt[k, :k] = 1.0
            rhs = np.r_[c[free] - q[np.ix_(free, fixed)] @ x[fixed], 1.0 - x.sum()]
            x[free] = np.linalg.solve(kkt, rhs)[:k]
        if abs(x.sum() - 1) > 1e-9 or x.min() < -1e-12 or x.max() > cap + 1e-12:
            continue
        value = 0.5 * x @ q @ x - c @ x
        if value < best_value:
            best_value, best_x = value, x
    assert best_x is not None
    return best_x


@settings(max_examples=120, deadline=None)
@given(n=st.integers(2, 6), seed=st.integers(0, 10**6), capped=st.booleans())
def test_matches_brute_force_enumeration(n: int, seed: int, capped: bool) -> None:
    q, c = random_problem(n, seed)
    cap = max(1.0 / n + 0.05, 0.35) if capped else 1.0
    sol = solve_simplex_qp(q, c, upper=cap)
    np.testing.assert_allclose(sol.x, brute_force(q, c, cap), atol=1e-9)
    assert sol.kkt_residual < 1e-9


@settings(max_examples=60, deadline=None)
@given(n=st.integers(2, 40), seed=st.integers(0, 10**6))
def test_kkt_conditions_hold_at_the_solution(n: int, seed: int) -> None:
    q, c = random_problem(n, seed)
    sol = solve_simplex_qp(q, c, upper=max(0.1, 1.0 / n))
    x, g, nu = sol.x, q @ sol.x - c, sol.multiplier
    assert x.sum() == pytest.approx(1.0)
    free = (x > 1e-10) & (x < max(0.1, 1.0 / n) - 1e-10)
    np.testing.assert_allclose(g[free], nu, atol=1e-9)
    assert np.all(g[x <= 1e-10] >= nu - 1e-9)  # raising a zero weight cannot help
    assert np.all(g[x >= max(0.1, 1.0 / n) - 1e-10] <= nu + 1e-9)  # nor cutting a capped one


def test_interior_solution_equals_the_closed_form() -> None:
    q = np.array([[0.04, 0.006], [0.006, 0.09]])
    closed = np.linalg.solve(q, np.ones(2))
    closed /= closed.sum()
    np.testing.assert_allclose(solve_simplex_qp(q).x, closed)


def test_corner_and_fully_capped_solutions() -> None:
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(solve_simplex_qp(q, [10.0, 0.0]).x, [1.0, 0.0])  # corner
    np.testing.assert_allclose(solve_simplex_qp(q, upper=0.5).x, [0.5, 0.5])  # no freedom
    lo = solve_simplex_qp(np.eye(3), [0.0, 0.0, 5.0], lower=[0.2, 0.3, 0.0])
    np.testing.assert_allclose(lo.x, [0.2, 0.3, 0.5])


def test_invalid_problems() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        solve_simplex_qp(np.eye(3), upper=0.2)
    with pytest.raises(ValueError, match="symmetric"):
        solve_simplex_qp(np.array([[1.0, 2.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="square"):
        solve_simplex_qp(np.ones((2, 3)))
    with pytest.raises(ValueError, match="lower <= upper"):
        solve_simplex_qp(np.eye(2), lower=0.6, upper=0.5)
