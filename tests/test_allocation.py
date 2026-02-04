"""Portfolio construction: IV, minimum variance, mean-variance, ERC and HRP."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester.portfolio import (
    ALLOCATORS,
    EqualRiskContribution,
    EqualWeight,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanVariance,
    MinimumVariance,
    SampleCovariance,
    correlation_distance,
    equal_risk_contribution_weights,
    hierarchical_risk_parity_weights,
    inverse_variance_weights,
    inverse_volatility_weights,
    mean_variance_weights,
    minimum_variance_weights,
    quasi_diagonal_order,
    risk_contributions,
    single_linkage,
)


def random_cov(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return (a @ a.T / n + np.diag(rng.uniform(0.01, 1.0, n))) * 1e-4


# Four assets: 0 and 2 correlated 0.8, 1 and 3 correlated 0.6, no correlation across.
VOLS = np.array([0.1, 0.2, 0.2, 0.1])
CORR = np.array(
    [
        [1.0, 0.0, 0.8, 0.0],
        [0.0, 1.0, 0.0, 0.6],
        [0.8, 0.0, 1.0, 0.0],
        [0.0, 0.6, 0.0, 1.0],
    ]
)
COV4 = CORR * np.outer(VOLS, VOLS)


class TestHRPHandWorked:
    """The 4-asset example worked by hand (see test docstrings for the arithmetic)."""

    def test_linkage_and_order(self) -> None:
        # d_02 = sqrt(0.1), d_13 = sqrt(0.2), the rest sqrt(0.5). Distances between the
        # columns of d: pairs (0,2) -> sqrt(2) d_02 = 0.4472, (1,3) -> sqrt(2) d_13 =
        # 0.6325, every cross pair -> 1.1046. Single linkage merges {0,2}, then {1,3},
        # then the two clusters; the leaf order is 0, 2, 1, 3.
        d = correlation_distance(COV4)
        assert d[0, 2] == pytest.approx(np.sqrt(0.1))
        euclid = np.sqrt(np.sum((d[:, :, None] - d[:, None, :]) ** 2, axis=0))
        z = single_linkage(euclid)
        np.testing.assert_allclose(z[:, [0, 1, 3]], [[0, 2, 2], [1, 3, 2], [4, 5, 4]])
        np.testing.assert_allclose(z[:2, 2], [np.sqrt(0.2), np.sqrt(0.4)])
        assert quasi_diagonal_order(z) == [0, 2, 1, 3]

    def test_weights(self) -> None:
        """Bisection of [0, 2 | 1, 3].

        Left cluster: inverse-variance weights (0.8, 0.2), variance
        0.64*0.01 + 0.04*0.04 + 2*0.8*0.2*0.016 = 0.01312. Right cluster: (0.2, 0.8) on
        (1, 3), variance 0.04*0.04 + 0.64*0.01 + 2*0.2*0.8*0.012 = 0.01184. So the left
        half gets alpha = 1 - 0.01312/0.02496 = 37/78 and the right 41/78. Inside the
        halves the single-asset variances split 0.8/0.2: w = (148, 41, 37, 164) / 390.
        """
        np.testing.assert_allclose(
            hierarchical_risk_parity_weights(COV4), np.array([148, 41, 37, 164]) / 390
        )


def test_single_linkage_on_a_line() -> None:
    # points 0, 1, 3, 7 on a line: merges (0,1)@1, then {0,1}+2 @2, then +3 @4
    pts = np.array([0.0, 1.0, 3.0, 7.0])
    z = single_linkage(np.abs(pts[:, None] - pts[None, :]))
    np.testing.assert_allclose(z, [[0, 1, 1, 2], [2, 4, 2, 3], [3, 5, 4, 4]])
    assert quasi_diagonal_order(z) == [3, 2, 0, 1]
    assert quasi_diagonal_order(single_linkage(np.zeros((1, 1)))) == [0]


@settings(max_examples=60, deadline=None)
@given(n=st.integers(2, 25), seed=st.integers(0, 10**6))
def test_hrp_is_a_positive_budget(n: int, seed: int) -> None:
    w = hierarchical_risk_parity_weights(random_cov(n, seed))
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_hrp_depends_on_the_labels_of_the_assets() -> None:
    """A property of the published algorithm, not of this implementation.

    A merge of two single assets lists the smaller label first, and the bisection cuts
    the leaf order by position rather than along the tree. When a cut falls between
    the two members of such a pair, relabelling them moves a different asset into each
    half, and the weights change (here by up to 14% relative).
    """
    cov = random_cov(6, 0)
    perm = np.random.default_rng(0).permutation(6)
    w = hierarchical_risk_parity_weights(cov)
    relabelled = hierarchical_risk_parity_weights(cov[np.ix_(perm, perm)])
    assert np.max(np.abs(relabelled / w[perm] - 1)) > 0.1
    # merges of a single asset with a cluster put the asset first whatever its label,
    # so with three assets any relabelling gives the same weights
    three = random_cov(3, 1)
    for p in ([1, 2, 0], [2, 0, 1], [0, 2, 1]):
        np.testing.assert_allclose(
            hierarchical_risk_parity_weights(three[np.ix_(p, p)]),
            hierarchical_risk_parity_weights(three)[p],
        )


def test_hrp_of_uncorrelated_assets_is_inverse_variance_for_two() -> None:
    cov = np.diag([0.01, 0.04])
    np.testing.assert_allclose(hierarchical_risk_parity_weights(cov), inverse_variance_weights(cov))


@settings(max_examples=80, deadline=None)
@given(n=st.integers(2, 40), seed=st.integers(0, 10**6))
def test_erc_equalises_risk_contributions(n: int, seed: int) -> None:
    cov = random_cov(n, seed)
    w = equal_risk_contribution_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(risk_contributions(w, cov), 1.0 / n, rtol=1e-8)


def test_erc_budgets_and_special_cases() -> None:
    cov = random_cov(5, 1)
    budgets = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
    np.testing.assert_allclose(
        risk_contributions(equal_risk_contribution_weights(cov, budgets), cov), budgets, rtol=1e-8
    )
    # uncorrelated assets, or two assets with any correlation: ERC is inverse volatility
    for cov2 in (np.diag([0.01, 0.04, 0.09]), np.array([[0.04, 0.03], [0.03, 0.09]])):
        np.testing.assert_allclose(
            equal_risk_contribution_weights(cov2), inverse_volatility_weights(cov2)
        )
    with pytest.raises(ValueError, match="budgets"):
        equal_risk_contribution_weights(cov, [1, 1, 1, 1, -1])


def test_minimum_and_mean_variance() -> None:
    cov = random_cov(6, 2)
    w = minimum_variance_weights(cov)
    rng = np.random.default_rng(3)
    for _ in range(200):  # no random long-only portfolio has lower variance
        other = rng.dirichlet(np.ones(6))
        assert w @ cov @ w <= other @ cov @ other + 1e-15
    capped = minimum_variance_weights(cov, max_weight=0.2)
    assert capped.max() <= 0.2 + 1e-12
    mu = np.array([0.05, 0.01, 0.02, 0.0, 0.03, 0.01]) / 252
    np.testing.assert_allclose(mean_variance_weights(mu, cov, risk_aversion=1e9), w, atol=1e-6)
    greedy = mean_variance_weights(mu, cov, risk_aversion=1e-6)
    assert int(np.argmax(greedy)) == 0
    assert greedy[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="max_weight"):
        minimum_variance_weights(cov, max_weight=0.1)
    with pytest.raises(ValueError, match="risk_aversion"):
        mean_variance_weights(mu, cov, risk_aversion=0)
    with pytest.raises(ValueError, match="positive variances"):
        inverse_volatility_weights(np.diag([0.01, 0.0]))


def test_allocators_on_return_windows() -> None:
    rng = np.random.default_rng(4)
    window = rng.normal(0.0003, 0.01, (252, 5)) * np.array([1, 2, 1, 3, 1])
    for name, cls in ALLOCATORS.items():
        w = cls()(window)
        assert w.shape == (5,), name
        assert w.sum() == pytest.approx(1.0), name
        assert (w >= -1e-12).all(), name
    np.testing.assert_allclose(EqualWeight()(window), np.full(5, 0.2))
    iv = InverseVolatility(covariance=SampleCovariance())(window)
    np.testing.assert_allclose(iv, inverse_volatility_weights(np.cov(window, rowvar=False)))
    assert MinimumVariance(max_weight=0.3)(window).max() <= 0.3 + 1e-12
    assert MeanVariance(risk_aversion=5.0).name == "meanvar"
    assert EqualRiskContribution()(window[:, :1]).tolist() == [1.0]
    assert set(ALLOCATORS) == {"ew", "iv", "minvar", "meanvar", "erc", "hrp"}
    assert HierarchicalRiskParity().name == "hrp"
    with pytest.raises(ValueError, match="N >= 1"):
        EqualWeight()(np.zeros((10, 0)))
