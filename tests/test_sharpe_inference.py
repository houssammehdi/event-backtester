"""Sharpe-ratio inference: standard errors, PSR, MinTRL (Bailey & Lopez de Prado, 2012)."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from backtester.validation import (
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sample_moments,
    sharpe_inference,
    sharpe_ratio_std_error,
)


def test_std_error_reduces_to_lo_for_normal_returns() -> None:
    sr, n = 0.08, 1000
    assert sharpe_ratio_std_error(sr, n) == pytest.approx(math.sqrt((1 + sr**2 / 2) / (n - 1)))
    # negative skew and fat tails widen it
    assert sharpe_ratio_std_error(sr, n, skew=-1.0, kurtosis=9.0) > sharpe_ratio_std_error(sr, n)


def test_min_track_record_hand_value() -> None:
    # SR = 0.1 per period against 0, normal returns, 95%: 1 + (1 + 0.005) * (1.6449 / 0.1)^2
    z = NormalDist().inv_cdf(0.95)
    expected = 1 + (1 + 0.5 * 0.1**2) * (z / 0.1) ** 2
    assert min_track_record_length(0.1) == pytest.approx(expected)
    assert min_track_record_length(0.1) == pytest.approx(273.0, abs=0.5)


def test_min_track_record_is_where_psr_reaches_the_confidence() -> None:
    """At exactly MinTRL observations the PSR equals the confidence level."""
    for sr, skew, kurt, conf in ((0.1, 0.0, 3.0, 0.95), (0.05, -0.8, 7.0, 0.9)):
        n = min_track_record_length(sr, 0.02, skew, kurt, conf)
        # PSR is continuous in n; evaluate it at the (fractional) MinTRL directly
        se = math.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
        assert NormalDist().cdf((sr - 0.02) / se) == pytest.approx(conf)
        assert probabilistic_sharpe_ratio(sr, 0.02, math.ceil(n), skew, kurt) >= conf


def test_min_track_record_edge_cases() -> None:
    assert min_track_record_length(0.05, benchmark_sharpe=0.05) == math.inf
    assert min_track_record_length(-0.1) == math.inf
    assert math.isnan(min_track_record_length(math.nan))
    with pytest.raises(ValueError, match="confidence"):
        min_track_record_length(0.1, confidence=1.0)


def test_sharpe_inference_collects_everything() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0006, 0.01, 1500)
    trials = rng.normal(0.0, 0.02, 30)
    info = sharpe_inference(r, periods_per_year=252, trial_sharpes=trials)
    sr, skew, kurt, n = sample_moments(r)
    assert info.n_obs == n == 1500
    assert info.sharpe == pytest.approx(sr)
    assert info.sharpe_annualized == pytest.approx(sr * math.sqrt(252))
    assert info.psr == pytest.approx(probabilistic_sharpe_ratio(sr, 0.0, n, skew, kurt))
    assert info.min_track_record_years == pytest.approx(
        min_track_record_length(sr, 0.0, skew, kurt) / 252
    )
    assert info.track_record_years == pytest.approx(1500 / 252)
    assert info.n_trials == 30
    assert info.dsr is not None
    assert info.dsr < info.psr  # deflation for 30 trials
    assert sharpe_inference(r).dsr is None
