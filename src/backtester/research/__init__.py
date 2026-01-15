"""Research tools: grid search, walk-forward optimisation, deflated Sharpe, fast path."""

from backtester.research.deflated import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sample_moments,
)
from backtester.research.grid import (
    MULTIPLE_TESTING_NOTE,
    GridSearchResult,
    expand_grid,
    grid_search,
    sharpe_objective,
)
from backtester.research.vectorized import VectorizedResult, run_vectorized
from backtester.research.walkforward import (
    WalkForwardResult,
    WalkForwardWindow,
    walk_forward,
    walk_forward_windows,
)

__all__ = [
    "MULTIPLE_TESTING_NOTE",
    "GridSearchResult",
    "VectorizedResult",
    "WalkForwardResult",
    "WalkForwardWindow",
    "deflated_sharpe_ratio",
    "expand_grid",
    "expected_max_sharpe",
    "grid_search",
    "probabilistic_sharpe_ratio",
    "run_vectorized",
    "sample_moments",
    "sharpe_objective",
    "walk_forward",
    "walk_forward_windows",
]
