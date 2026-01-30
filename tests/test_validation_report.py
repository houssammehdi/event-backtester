"""The one-call validation reports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtester.validation import validate_family, validate_returns


def test_validate_returns_report() -> None:
    r = np.random.default_rng(0).normal(0.0006, 0.01, 1500)
    v = validate_returns(r, n_samples=300, trial_sharpes=[0.01, 0.05, 0.06])
    text = v.format()
    for needle in ("Sharpe ratio (annualised)", "PSR", "Min. track record", "Deflated Sharpe (3"):
        assert needle in text
    assert v.bootstrap.n_samples == 300
    assert v.sharpe.dsr is not None


def test_validate_family_report() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(0.0, 0.01, (1300, 6)), columns=[f"p={j}" for j in range(6)])
    frame["p=4"] += 0.001
    fam = validate_family(frame, n_samples=300, pbo_splits=8, lookback=10)
    assert fam.best_label == "p=4"
    assert fam.pbo is not None
    assert fam.pbo.n_splits == 8
    assert fam.cpcv is not None
    assert fam.cpcv.n_paths == 5
    text = fam.format()
    for needle in (
        "Selected: p=4",
        "Probability of backtest overfitting",
        "White's Reality Check",
        "Hansen SPA",
        "Combinatorial purged CV",
        "within 10 bars",
    ):
        assert needle in text


def test_validate_family_single_configuration_and_short_samples() -> None:
    one = pd.DataFrame({"only": np.random.default_rng(2).normal(0, 0.01, 40)})
    fam = validate_family(one, n_samples=100, pbo_splits=16)
    assert fam.pbo is None
    assert fam.cpcv is None
    two = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (40, 2)))
    assert validate_family(two, n_samples=100, pbo_splits=64).pbo is not None  # splits shrunk
    with pytest.raises(ValueError, match="at least one"):
        validate_family(pd.DataFrame(index=range(40)))
