"""Statistical validation of backtests: uncertainty, overfitting and data snooping.

A backtest is one draw of history, usually the best of many tried. This package
quantifies how much of its performance could be luck:

* **Sharpe-ratio inference** (:mod:`~backtester.validation.sharpe`) - probabilistic
  and deflated Sharpe ratio, minimum track record length, standard errors under
  skewness and fat tails.
* **Stationary bootstrap** (:mod:`~backtester.validation.bootstrap`) - confidence
  intervals for Sharpe, CAGR, drawdown and more, with automatic block-length
  selection.
* **Probability of backtest overfitting** (:mod:`~backtester.validation.pbo`) -
  combinatorially symmetric cross-validation of a parameter search.
* **Data-snooping tests** (:mod:`~backtester.validation.spa`) - White's Reality Check
  and Hansen's test for superior predictive ability over a strategy family.
* **Purged cross-validation** (:mod:`~backtester.validation.cv`) - purged K-fold and
  combinatorial purged CV with embargo, including CPCV backtest paths.
"""

from backtester.validation.bootstrap import (
    BlockLength,
    BootstrapResult,
    Interval,
    bootstrap_statistics,
    default_statistics,
    optimal_block_length,
    resample_counts,
    stationary_bootstrap_indices,
)
from backtester.validation.cv import (
    CombinatorialPurgedCV,
    CPCVResult,
    CVSplit,
    PurgedKFold,
    cpcv_backtest,
    label_spans,
    purged_train_indices,
)
from backtester.validation.pbo import PBOResult, probability_of_backtest_overfitting
from backtester.validation.sharpe import (
    SharpeInference,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sample_moments,
    sharpe_inference,
    sharpe_ratio_std_error,
)
from backtester.validation.spa import SPAResult, long_run_variance, superior_predictive_ability

__all__ = [
    "BlockLength",
    "BootstrapResult",
    "CPCVResult",
    "CVSplit",
    "CombinatorialPurgedCV",
    "Interval",
    "PBOResult",
    "PurgedKFold",
    "SPAResult",
    "SharpeInference",
    "bootstrap_statistics",
    "cpcv_backtest",
    "default_statistics",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "label_spans",
    "long_run_variance",
    "min_track_record_length",
    "optimal_block_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_train_indices",
    "resample_counts",
    "sample_moments",
    "sharpe_inference",
    "sharpe_ratio_std_error",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
]
