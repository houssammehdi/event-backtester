"""One-call validation of a strategy, or of a parameter search, with a text report.

* :func:`validate_returns` - how uncertain is one strategy's performance? Stationary
  bootstrap intervals for Sharpe, CAGR, drawdown and volatility, plus the PSR, the
  minimum track record length and (given the trials of a search) the deflated Sharpe.
* :func:`validate_family` - how much of a parameter search's best result is selection?
  The above for the selected configuration, plus the probability of backtest
  overfitting (CSCV), White's Reality Check and Hansen's SPA against a benchmark, and
  the distribution of out-of-sample Sharpe ratios over CPCV backtest paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.analytics.report import num, pct, table
from backtester.validation.bootstrap import BootstrapResult, Seed, bootstrap_statistics
from backtester.validation.cv import CombinatorialPurgedCV, CPCVResult, cpcv_backtest
from backtester.validation.pbo import PBOResult, probability_of_backtest_overfitting
from backtester.validation.sharpe import SharpeInference, sample_moments, sharpe_inference
from backtester.validation.spa import SPAResult, superior_predictive_ability

_BOOTSTRAPPED = (
    ("sharpe", "Sharpe ratio (annualised)", num),
    ("cagr", "CAGR", pct),
    ("max_drawdown", "Max drawdown", pct),
    ("volatility", "Annual volatility", pct),
)


def _years(value: float) -> str:
    if math.isinf(value):
        return "never"
    return "n/a" if math.isnan(value) else f"{value:.1f} years"


@dataclass(frozen=True, slots=True)
class StrategyValidation:
    """Output of :func:`validate_returns`."""

    sharpe: SharpeInference
    bootstrap: BootstrapResult

    def format(self, title: str = "Performance with uncertainty") -> str:
        """Text table of the estimates and their bootstrap intervals."""
        level = f"{self.bootstrap.confidence:.0%} interval"
        rows = []
        for key, label, fmt in _BOOTSTRAPPED:
            iv = self.bootstrap.intervals[key]
            rows.append((label, fmt(iv.estimate), f"{fmt(iv.lower)} .. {fmt(iv.upper)}"))
        s = self.sharpe
        rows.append(("PSR (true Sharpe > 0)", num(s.psr, 3), ""))
        rows.append(
            (
                f"Min. track record ({s.confidence:.0%})",
                _years(s.min_track_record_years),
                f"(sample: {_years(s.track_record_years)})",
            )
        )
        if s.dsr is not None:
            rows.append((f"Deflated Sharpe ({s.n_trials} trials)", num(s.dsr, 3), ""))
        lines = [table(rows, header=(title, "estimate", level))]
        lines.append(
            f"Stationary bootstrap: {self.bootstrap.n_samples} resamples, mean block length "
            f"{self.bootstrap.block_length:.1f} bars; PSR and MinTRL adjust for skew "
            f"{s.skew:.2f} and kurtosis {s.kurtosis:.1f}."
        )
        return "\n".join(lines)


def validate_returns(
    returns: npt.ArrayLike | pd.Series,
    *,
    periods_per_year: float = 252.0,
    n_samples: int = 2000,
    block_length: float | None = None,
    confidence: float = 0.95,
    benchmark_sharpe: float = 0.0,
    trial_sharpes: npt.ArrayLike | None = None,
    seed: Seed = 0,
) -> StrategyValidation:
    """Bootstrap intervals and Sharpe-ratio inference for one return series.

    Args:
        returns: Per-period returns (e.g. :attr:`BacktestResult.returns`).
        periods_per_year: Annualisation.
        n_samples: Bootstrap resamples.
        block_length: Stationary-bootstrap mean block length (``None``: Politis-White).
        confidence: Interval coverage and MinTRL confidence.
        benchmark_sharpe: Per-period Sharpe ratio the PSR and MinTRL test against.
        trial_sharpes: Per-period Sharpe ratios of all configurations of the search
            that produced ``returns`` (adds the deflated Sharpe ratio).
        seed: Bootstrap seed.
    """
    r = np.asarray(returns, dtype=np.float64)
    return StrategyValidation(
        sharpe=sharpe_inference(
            r,
            benchmark_sharpe=benchmark_sharpe,
            periods_per_year=periods_per_year,
            confidence=confidence,
            trial_sharpes=trial_sharpes,
        ),
        bootstrap=bootstrap_statistics(
            r,
            n_samples=n_samples,
            block_length=block_length,
            confidence=confidence,
            periods_per_year=periods_per_year,
            seed=seed,
        ),
    )


@dataclass(frozen=True, slots=True)
class FamilyValidation:
    """Output of :func:`validate_family`."""

    labels: tuple[str, ...]
    best: int
    """Column of the configuration with the best full-sample Sharpe ratio."""
    selected: StrategyValidation
    pbo: PBOResult | None
    spa: SPAResult
    cpcv: CPCVResult | None
    benchmark_name: str
    lookback: int = 0
    """Look-back used to purge the CPCV training sets."""

    @property
    def best_label(self) -> str:
        """Label of the selected configuration."""
        return self.labels[self.best]

    def format(self) -> str:
        """Multi-section text report."""
        n = len(self.labels)
        lines = [
            f"Selected: {self.best_label}  (best full-sample Sharpe of {n} configurations)",
            "",
            self.selected.format("Selected configuration"),
            "",
        ]
        if self.pbo is not None:
            p = self.pbo
            rows = [
                ("Probability of backtest overfitting", pct(p.pbo, 1)),
                ("Degradation slope (OOS on IS Sharpe)", num(p.degradation_slope)),
                ("P(selected loses out of sample)", pct(p.prob_oos_loss, 1)),
            ]
            lines.append(table(rows, header=("Backtest overfitting (CSCV)", "")))
            lines.append(
                f"{p.n_splits} blocks of {p.n_obs // p.n_splits} bars, "
                f"{p.n_combinations} train/test combinations."
            )
            lines.append("")
        s = self.spa
        rows = [
            ("White's Reality Check", num(s.reality_check_pvalue, 3)),
            ("Hansen SPA (consistent)", num(s.pvalue_consistent, 3)),
            (
                "Hansen SPA (lower / upper)",
                f"{num(s.pvalue_lower, 3)} / {num(s.pvalue_upper, 3)}",
            ),
        ]
        lines.append(table(rows, header=(f"Data snooping vs {self.benchmark_name}", "p-value")))
        lines.append(
            f"H0: none of the {n} configurations beats {self.benchmark_name}; "
            f"{s.n_samples} resamples, block length {s.block_length:.1f} bars."
        )
        if self.cpcv is not None:
            c = self.cpcv
            sr = c.path_sharpe
            rows = [
                ("Backtest paths", str(c.n_paths)),
                (
                    "OOS Sharpe: mean / min / max",
                    f"{num(float(np.mean(sr)))} / {num(float(np.min(sr)))} / "
                    f"{num(float(np.max(sr)))}",
                ),
            ]
            lines += ["", table(rows, header=("Combinatorial purged CV", ""))]
            lines.append(
                f"{c.n_groups} groups, {c.n_test_groups} per test set; training returns "
                f"within {self.lookback} bars of a test block purged, embargo {c.embargo} bars."
            )
        return "\n".join(lines)


def _largest_even_split(n_obs: int, requested: int) -> int:
    splits = min(requested, n_obs // 2)
    return splits - splits % 2


def validate_family(
    returns: pd.DataFrame,
    benchmark: pd.Series | npt.ArrayLike | None = None,
    *,
    benchmark_name: str = "cash",
    periods_per_year: float = 252.0,
    n_samples: int = 2000,
    confidence: float = 0.95,
    pbo_splits: int = 16,
    cpcv: CombinatorialPurgedCV | None = None,
    lookback: int = 0,
    seed: Seed = 0,
) -> FamilyValidation:
    """Validate a parameter search from its performance matrix.

    Args:
        returns: ``(T, N)`` per-period returns, one column per configuration, all over
            the same periods (e.g. :attr:`GridSearchResult.returns`).
        benchmark: Benchmark returns for the SPA / Reality Check (``None``: zero).
        benchmark_name: How the benchmark is named in the report.
        periods_per_year: Annualisation.
        n_samples: Bootstrap resamples (intervals and SPA).
        confidence: Interval coverage.
        pbo_splits: CSCV blocks (reduced to fit short samples).
        cpcv: CPCV splitter for the path distribution (default: 6 groups, 2 test
            groups, no embargo); skipped for a single configuration.
        lookback: Bars of history behind each return (the strategies' look-back). The
            return at bar ``i`` is given the information interval ``[i - lookback, i]``,
            so CPCV purges training returns within ``lookback`` bars of a test block,
            whose positions were formed from test-period prices (or vice versa).
        seed: Seed of every bootstrap.
    """
    if returns.shape[1] < 1:
        raise ValueError("need at least one configuration")
    values = returns.to_numpy(dtype=np.float64)
    labels = tuple(str(c) for c in returns.columns)
    trial_sharpes = np.array([sample_moments(values[:, j])[0] for j in range(values.shape[1])])
    best = int(np.argmax(np.where(np.isnan(trial_sharpes), -np.inf, trial_sharpes)))
    selected = validate_returns(
        values[:, best],
        periods_per_year=periods_per_year,
        n_samples=n_samples,
        confidence=confidence,
        trial_sharpes=trial_sharpes,
        seed=seed,
    )
    n_obs, n_configs = values.shape
    pbo = None
    splits = _largest_even_split(n_obs, pbo_splits)
    if n_configs >= 2 and splits >= 2:
        pbo = probability_of_backtest_overfitting(returns, n_splits=splits)
    spa = superior_predictive_ability(
        returns,
        None if benchmark is None else np.asarray(benchmark, dtype=np.float64),
        n_samples=n_samples,
        seed=seed,
    )
    cv_result = None
    if n_configs >= 2:
        splitter = cpcv or CombinatorialPurgedCV(6, 2)
        start = np.arange(n_obs) - lookback
        cv_result = cpcv_backtest(
            returns, splitter, start=start, end=np.arange(n_obs), periods_per_year=periods_per_year
        )
    return FamilyValidation(
        labels=labels,
        best=best,
        selected=selected,
        pbo=pbo,
        spa=spa,
        cpcv=cv_result,
        benchmark_name=benchmark_name,
        lookback=lookback,
    )
