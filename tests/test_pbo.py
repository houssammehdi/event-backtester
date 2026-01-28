"""Probability of backtest overfitting via CSCV (Bailey, Borwein, Lopez de Prado & Zhu)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from backtester.validation import probability_of_backtest_overfitting


def brute_force_logits(r: np.ndarray, n_splits: int) -> list[float]:
    """Direct transcription of the CSCV algorithm, recomputing every Sharpe ratio."""
    t, n = r.shape
    rows = t // n_splits
    r = r[t - rows * n_splits :]
    blocks = np.split(np.arange(len(r)), n_splits)
    logits = []
    for chosen in itertools.combinations(range(n_splits), n_splits // 2):
        train = np.concatenate([blocks[b] for b in chosen])
        test = np.setdiff1d(np.arange(len(r)), train)
        sr_train = r[train].mean(0) / r[train].std(0, ddof=1)
        sr_test = r[test].mean(0) / r[test].std(0, ddof=1)
        best = int(np.argmax(sr_train))
        rank = 1 + np.sum(sr_test < sr_test[best])
        w = rank / (n + 1)
        logits.append(math.log(w / (1 - w)))
    return logits


@pytest.mark.parametrize(("t", "n", "s"), [(120, 7, 6), (203, 5, 8), (64, 12, 4)])
def test_block_sums_match_the_direct_algorithm(t: int, n: int, s: int) -> None:
    r = np.random.default_rng(t).normal(0.0003, 0.01, (t, n))
    res = probability_of_backtest_overfitting(r, n_splits=s)
    np.testing.assert_allclose(res.logits, brute_force_logits(r, s), atol=1e-10)
    assert res.n_combinations == math.comb(s, s // 2)
    assert res.n_obs == (t // s) * s  # the oldest t mod s rows are dropped


def test_pure_noise_gives_pbo_one_half() -> None:
    """With N even and iid noise the selected config's OOS rank is uniform: E[PBO] = 1/2."""
    rng = np.random.default_rng(0)
    values = [
        probability_of_backtest_overfitting(rng.standard_normal((600, 20)), n_splits=8).pbo
        for _ in range(120)
    ]
    assert np.mean(values) == pytest.approx(0.5, abs=0.05)
    # the OOS relative rank of the IS winner is uniform too: mean logit ~ 0
    res = probability_of_backtest_overfitting(rng.standard_normal((2000, 40)), n_splits=10)
    assert np.mean(res.relative_ranks) == pytest.approx(0.5, abs=0.12)


def test_noise_shows_performance_degradation() -> None:
    """Complementary halves of noise: a high IS Sharpe predicts a low OOS Sharpe."""
    res = probability_of_backtest_overfitting(
        np.random.default_rng(1).standard_normal((1000, 30)), n_splits=10
    )
    assert res.degradation_slope < 0


def test_a_genuinely_skilled_strategy_is_not_overfit() -> None:
    """One strategy with real skill among 19 noise rivals is selected and holds up OOS.

    (With half the data, or a per-period Sharpe of 0.15, detection is much less
    reliable: over 40 seeds the PBO then reaches 0.33. See docs/methodology.md.)
    """
    rng = np.random.default_rng(2)
    r = rng.standard_normal((2000, 20))
    r[:, 3] += 0.2  # per-period Sharpe 0.2 (about 3.2 annualised for daily data)
    res = probability_of_backtest_overfitting(r, n_splits=10)
    assert res.pbo < 0.02
    assert res.selection_frequency().idxmax() == "3"
    assert res.prob_oos_loss < 0.05
    assert res.selection_frequency().sum() == pytest.approx(1.0)


def test_labels_ties_and_metrics() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 0.01, (400, 3))
    frame = pd.DataFrame(np.column_stack([base, base[:, 0]]), columns=["a", "b", "c", "a_copy"])
    res = probability_of_backtest_overfitting(frame, n_splits=4)
    assert res.labels == ("a", "b", "c", "a_copy")
    # "a" and its copy tie everywhere: the winner is the first column among equals and
    # ties share the average rank, so the copy is never selected
    assert "a_copy" not in {res.labels[i] for i in res.selected}
    mean_builtin = probability_of_backtest_overfitting(frame, n_splits=4, metric="mean")
    mean_custom = probability_of_backtest_overfitting(
        frame, n_splits=4, metric=lambda block: block.mean(axis=0)
    )
    np.testing.assert_allclose(mean_builtin.logits, mean_custom.logits)
    assert mean_custom.metric == "<lambda>"
    sortino = probability_of_backtest_overfitting(frame, n_splits=4, metric="sortino")
    assert 0.0 <= sortino.pbo <= 1.0


def test_flat_configuration_has_zero_sharpe_not_nan() -> None:
    r = np.random.default_rng(4).normal(0.0005, 0.01, (300, 3))
    r[:, 1] = 0.0  # a configuration that never trades
    res = probability_of_backtest_overfitting(r, n_splits=6)
    assert np.isfinite(res.logits).all()


def test_invalid_inputs() -> None:
    r = np.zeros((100, 3))
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(r, n_splits=5)
    with pytest.raises(ValueError, match="two"):
        probability_of_backtest_overfitting(np.zeros((100, 1)), n_splits=4)
    with pytest.raises(ValueError, match="periods"):
        probability_of_backtest_overfitting(np.zeros((10, 3)), n_splits=8)
    with pytest.raises(ValueError, match="NaN"):
        probability_of_backtest_overfitting(np.full((100, 3), np.nan), n_splits=4)
    with pytest.raises(ValueError, match="metric"):
        probability_of_backtest_overfitting(r, n_splits=4, metric="calmar")  # type: ignore[arg-type]
