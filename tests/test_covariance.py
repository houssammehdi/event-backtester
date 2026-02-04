"""Sample and Ledoit-Wolf covariance estimators against direct computations."""

from __future__ import annotations

import numpy as np
import pytest

from backtester.portfolio import LedoitWolf, SampleCovariance, ledoit_wolf, sample_covariance


def returns(t: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(n, n)) / np.sqrt(n)
    return (rng.standard_t(5, size=(t, n)) @ mix.T + rng.normal(size=(t, 1))) * 0.01


def reference_constant_correlation(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004a), transcribed entry by entry (loops, no matrix algebra).

    pi_ij = 1/T sum_t (y_ti y_tj - s_ij)^2
    theta_ii,ij = 1/T sum_t (y_ti^2 - s_ii) (y_ti y_tj - s_ij)
    rho = sum_i pi_ii + sum_{i != j} r_bar / 2 (sqrt(s_jj / s_ii) theta_ii,ij
                                                + sqrt(s_ii / s_jj) theta_jj,ij)
    gamma = sum_ij (f_ij - s_ij)^2,  delta = max(0, min(1, (pi - rho) / gamma / T))
    """
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s[i, j] = sum(y[k, i] * y[k, j] for k in range(t)) / t
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    r_bar = sum(s[i, j] / np.sqrt(s[i, i] * s[j, j]) for i, j in pairs) / (n * (n - 1))
    f = np.diag(np.diag(s))
    for i, j in pairs:
        f[i, j] = r_bar * np.sqrt(s[i, i] * s[j, j])
    pi = np.zeros((n, n))
    theta = np.zeros((n, n))  # theta[i, j] = theta_ii,ij
    for i in range(n):
        for j in range(n):
            pi[i, j] = sum((y[k, i] * y[k, j] - s[i, j]) ** 2 for k in range(t)) / t
            theta[i, j] = (
                sum((y[k, i] ** 2 - s[i, i]) * (y[k, i] * y[k, j] - s[i, j]) for k in range(t)) / t
            )
    rho = sum(pi[i, i] for i in range(n))
    for i, j in pairs:
        rho += r_bar / 2 * np.sqrt(s[j, j] / s[i, i]) * theta[i, j]
        rho += r_bar / 2 * np.sqrt(s[i, i] / s[j, j]) * theta[j, i]
    gamma = sum((f[i, j] - s[i, j]) ** 2 for i in range(n) for j in range(n))
    delta = max(0.0, min(1.0, (pi.sum() - rho) / gamma / t))
    return delta * f + (1 - delta) * s, delta


def reference_identity(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004b) in the paper's notation: m, d^2, b-bar^2, b^2, a^2."""
    t, n = x.shape
    y = x - x.mean(axis=0)
    s = y.T @ y / t
    norm2 = lambda a: np.trace(a @ a.T) / n  # noqa: E731 - the paper's normalised norm
    m = np.trace(s) / n
    d2 = norm2(s - m * np.eye(n))
    b_bar2 = sum(norm2(np.outer(y[k], y[k]) - s) for k in range(t)) / t**2
    b2 = min(b_bar2, d2)
    a2 = d2 - b2
    return b2 / d2 * m * np.eye(n) + a2 / d2 * s, b2 / d2


@pytest.mark.parametrize(("t", "n", "seed"), [(40, 6, 0), (120, 4, 1), (15, 8, 2)])
def test_constant_correlation_target_matches_the_paper(t: int, n: int, seed: int) -> None:
    x = returns(t, n, seed)
    expected, delta = reference_constant_correlation(x)
    est = ledoit_wolf(x, "constant_correlation")
    assert est.shrinkage == pytest.approx(delta, abs=1e-12)
    np.testing.assert_allclose(est.covariance, expected, rtol=1e-12, atol=1e-18)
    assert 0 < est.shrinkage < 1


@pytest.mark.parametrize(("t", "n", "seed"), [(40, 6, 3), (200, 5, 4), (10, 12, 5)])
def test_identity_target_matches_the_paper(t: int, n: int, seed: int) -> None:
    x = returns(t, n, seed)
    expected, delta = reference_identity(x)
    est = ledoit_wolf(x, "identity")
    assert est.shrinkage == pytest.approx(delta, abs=1e-12)
    np.testing.assert_allclose(est.covariance, expected, rtol=1e-12, atol=1e-18)


def test_shrinkage_improves_conditioning_when_n_is_close_to_t() -> None:
    x = returns(30, 25, 6)
    sample = sample_covariance(x)
    for target in ("constant_correlation", "identity"):
        shrunk = ledoit_wolf(x, target).covariance  # type: ignore[arg-type]
        assert np.linalg.eigvalsh(shrunk).min() > 0
        assert np.linalg.cond(shrunk) < np.linalg.cond(sample) / 10


def test_more_data_means_less_shrinkage() -> None:
    small = ledoit_wolf(returns(50, 5, 7)).shrinkage
    large = ledoit_wolf(returns(5000, 5, 7)).shrinkage
    assert large < small


def test_fixed_intensity_estimators_and_errors() -> None:
    x = returns(60, 4, 8)
    half = ledoit_wolf(x, shrinkage=0.5)
    np.testing.assert_allclose(half.covariance, 0.5 * half.target + 0.5 * half.sample)
    np.testing.assert_allclose(sample_covariance(x), np.cov(x, rowvar=False))
    np.testing.assert_allclose(SampleCovariance()(x), np.cov(x, rowvar=False))
    np.testing.assert_allclose(LedoitWolf("identity")(x), ledoit_wolf(x, "identity").covariance)
    with pytest.raises(ValueError, match="target"):
        ledoit_wolf(x, "diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shrinkage"):
        ledoit_wolf(x, shrinkage=1.5)
    with pytest.raises(ValueError, match="finite"):
        ledoit_wolf(np.full((5, 2), np.nan))
    with pytest.raises(ValueError, match="2-D"):
        sample_covariance(np.zeros(5))
