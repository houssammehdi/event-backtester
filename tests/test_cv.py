"""Purged K-fold and combinatorial purged cross-validation (Lopez de Prado, 2018)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backtester.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    cpcv_backtest,
    label_spans,
    purged_train_indices,
)


def test_purging_removes_exactly_the_overlapping_labels() -> None:
    """Labels [i, i + 5]; test fold 40..59 spans [40, 64].

    Overlapping training labels start in 35..39 (end >= 40) or in 60..64 (start <= 64).
    An embargo of 3 then drops 65, 66 and 67.
    """
    start, end = label_spans(100, horizon=5)
    test = np.arange(40, 60)
    no_embargo = purged_train_indices(start, end, test)
    np.testing.assert_array_equal(no_embargo, np.r_[0:35, 65:100])
    embargoed = purged_train_indices(start, end, test, embargo=3)
    np.testing.assert_array_equal(embargoed, np.r_[0:35, 68:100])


def test_point_labels_need_no_purging_and_lookback_purges_both_sides() -> None:
    start, end = label_spans(50)
    np.testing.assert_array_equal(
        purged_train_indices(start, end, np.arange(20, 30)), np.r_[0:20, 30:50]
    )
    start, end = label_spans(50, lookback=4)  # [i - 4, i]
    np.testing.assert_array_equal(
        purged_train_indices(start, end, np.arange(20, 30)), np.r_[0:16, 34:50]
    )


def test_datetime_spans() -> None:
    days = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    ends = np.r_[days[2:], np.repeat(days[-1], 2)]  # each label ends two bars later
    train = purged_train_indices(days, ends, np.arange(10, 20))
    np.testing.assert_array_equal(train, np.r_[0:8, 22:30])


@settings(max_examples=150, deadline=None)
@given(
    n=st.integers(10, 120),
    horizons=st.lists(st.integers(0, 12), min_size=120, max_size=120),
    groups=st.integers(2, 6),
    data=st.data(),
    embargo=st.integers(0, 6),
)
def test_purge_is_sound_and_minimal(
    n: int, horizons: list[int], groups: int, data: st.DataObject, embargo: int
) -> None:
    start = np.arange(n)
    end = start + np.array(horizons[:n])
    chunks = np.array_split(np.arange(n), groups)
    chosen = data.draw(st.sets(st.integers(0, groups - 1), min_size=1, max_size=groups - 1))
    test = np.sort(np.concatenate([chunks[g] for g in chosen]))
    train = purged_train_indices(start, end, test, embargo=embargo)
    assert not set(train) & set(test)
    runs = np.split(test, np.flatnonzero(np.diff(test) > 1) + 1)
    spans = [(start[r[0]], end[r].max()) for r in runs]
    for i in train:  # sound: no training label overlaps a test block
        assert all(end[i] < lo or start[i] > hi for lo, hi in spans)
    for i in sorted(set(range(n)) - set(train) - set(test)):  # minimal: every drop is justified
        overlaps = any(start[i] <= hi and end[i] >= lo for lo, hi in spans)
        embargoed = any(
            0 <= i - int(np.searchsorted(start, hi, side="right")) < embargo for _, hi in spans
        )
        assert overlaps or embargoed


def test_purged_kfold_folds_and_embargo() -> None:
    splits = PurgedKFold(3, embargo=2).split(*label_spans(30, horizon=1))
    assert [s.test_groups for s in splits] == [(0,), (1,), (2,)]
    np.testing.assert_array_equal(splits[0].train, np.r_[13:30])  # 10 purged, 11-12 embargoed
    np.testing.assert_array_equal(splits[1].train, np.r_[0:9, 23:30])
    np.testing.assert_array_equal(splits[2].train, np.r_[0:19])
    with pytest.raises(ValueError, match="at least 2"):
        PurgedKFold(1)


class TestCPCV:
    def test_counts(self) -> None:
        for n_groups, k in ((6, 2), (8, 3), (5, 1)):
            cv = CombinatorialPurgedCV(n_groups, k)
            assert cv.n_splits == math.comb(n_groups, k)
            assert cv.n_paths == k * cv.n_splits // n_groups
            table = cv.path_splits()
            assert table.shape == (cv.n_paths, n_groups)
            # every split feeds each of its test groups into exactly one path
            for s, chosen in enumerate(cv.combinations()):
                for g in range(n_groups):
                    assert (table[:, g] == s).sum() == (g in chosen)

    def test_paths_are_complete_and_reuse_split_predictions(self) -> None:
        cv = CombinatorialPurgedCV(6, 2, embargo=1)
        splits = cv.split(*label_spans(61, horizon=2))
        assert len(splits) == 15
        for sp in splits:
            assert not set(sp.train) & set(sp.test)
        paths = cv.assemble_paths(61, [sp.test.astype(float) for sp in splits])
        assert paths.shape == (5, 61)
        for p in paths:  # each path covers every observation exactly once
            np.testing.assert_array_equal(p, np.arange(61))
        tagged = cv.assemble_paths(61, [np.full(len(sp.test), s) for s, sp in enumerate(splits)])
        groups = cv.groups(61)
        for p, row in enumerate(cv.path_splits()):
            for g, s in enumerate(row):
                assert (tagged[p, groups[g]] == s).all()

    def test_adjacent_test_groups_form_one_block(self) -> None:
        cv = CombinatorialPurgedCV(4, 2)
        start, end = label_spans(40, horizon=3)
        split = cv.split(start, end)[3]  # groups (1, 2): positions 10..29 as one block
        assert split.test_groups == (1, 2)
        np.testing.assert_array_equal(split.train, np.r_[0:7, 33:40])

    def test_invalid(self) -> None:
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(4, 4)
        with pytest.raises(ValueError, match="15 prediction arrays"):
            CombinatorialPurgedCV(6, 2).assemble_paths(10, [np.zeros(2)])
        with pytest.raises(ValueError, match="sorted"):
            purged_train_indices(np.array([3, 1, 2]), None, np.array([0]))


def test_cpcv_backtest_selects_the_skilled_configuration() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, (1200, 8)), columns=[f"c{j}" for j in range(8)])
    returns["c5"] += 0.0015  # per-period Sharpe 0.15
    res = cpcv_backtest(returns, CombinatorialPurgedCV(6, 2, embargo=5))
    assert set(res.selected) == {"c5"}
    assert res.n_paths == 5
    assert res.path_returns.shape == (1200, 5)
    np.testing.assert_allclose(res.path_returns["path_0"], returns["c5"])
    assert (res.path_sharpe > 1.5).all()


def test_cpcv_backtest_on_noise_has_paths_around_zero() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, (1500, 20)))
    res = cpcv_backtest(returns, CombinatorialPurgedCV(6, 2))
    assert abs(float(np.mean(res.path_sharpe))) < 0.8  # annualised; selection adds nothing
    with pytest.raises(ValueError, match="NaN"):
        cpcv_backtest(returns.where(returns > 0))
