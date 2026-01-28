"""Statistical validation of backtests: uncertainty, overfitting and data snooping.

A backtest is one draw of history, usually the best of many tried. This package
quantifies how much of its performance could be luck:

* **Sharpe-ratio inference** (:mod:`~backtester.validation.sharpe`) - probabilistic
  and deflated Sharpe ratio, minimum track record length, standard errors under
  skewness and fat tails.
"""

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

__all__ = [
    "SharpeInference",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "sample_moments",
    "sharpe_inference",
    "sharpe_ratio_std_error",
]
