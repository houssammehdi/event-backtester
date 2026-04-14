# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-25

### Added

- **Validation toolkit**, `backtester.validation`:
  - probability of backtest overfitting by combinatorially symmetric cross-validation
    (logit distribution, performance degradation, probability of loss);
  - the stationary bootstrap with automatic block length (Politis & White, with the
    Patton, Politis & White correction) and intervals for the Sharpe ratio, CAGR,
    maximum drawdown, volatility and any user statistic;
  - White's Reality Check and Hansen's SPA test (consistent, lower and upper
    p-values);
  - purged K-fold and combinatorial purged cross-validation with embargo, including
    the CPCV backtest paths and parameter selection inside them;
  - the probabilistic Sharpe ratio and minimum track record length next to the
    deflated Sharpe ratio, with standard errors under skewness and fat tails;
  - `validate_returns` and `validate_family`, one-call reports, also available as
    `GridSearchResult.validate()` and `WalkForwardResult.validate()`.
- **Portfolio construction**, `backtester.portfolio`: sample and Ledoit-Wolf
  covariance (constant-correlation and identity targets), inverse volatility,
  long-only minimum variance and mean-variance (exact active-set solver on the capped
  simplex), equal risk contribution (Spinu's convex formulation), hierarchical risk
  parity; `AllocationStrategy` runs any of them on a rebalancing schedule.
- **Order types**: brackets (entry, stop-loss, take-profit or trailing exit), OCO
  groups, trailing stops (absolute or percent), market-on-open and market-on-close.
  Orders execute along an explicit intrabar path; when the path matters the broker
  takes the worst case by default (`intrabar="worst" | "best" | "high_first" |
  "low_first"`).
- **Parallel research**: `grid_search(..., n_jobs=k)` and `walk_forward(..., n_jobs=k)`
  run in a process pool with results identical to the serial ones.
- **HTML tear sheet**: `BacktestResult.tear_sheet(path)` writes one self-contained
  file with equity, drawdown, rolling Sharpe and volatility, monthly returns, return
  and trade distributions, exposure, metrics and the validation statistics.
- **CLI**: `backtest validate`; `--report` for `run` and `validate`; `--jobs` for
  `validate` and `walkforward`; `--universe multi`; `--samples` and `--boot-seed`; the
  walk-forward report adds the in-sample PBO of every window and bootstrap intervals
  for the stitched out-of-sample curve.
- `generate_multi_asset`: a synthetic universe of equities, bonds and commodities with
  independent factors per asset class.
- `MarketView.previous_timestamp`, `orders.AUCTION_TYPES`, `DataFeed` pickling.
- Scripts: engine and parallel benchmarks, golden-output digests, the calibration
  study of the validation statistics, the docs screenshot. Examples: an allocation
  study with out-of-sample validation, and the intrabar sensitivity of a bracket
  strategy.
- Documentation: `docs/methodology.md` (execution model, bias taxonomy, validation
  statistics, calibration, references) and `docs/guide.md`.
- CI: tear-sheet and `backtest validate` smoke tests, and a job for the slow
  Monte Carlo tests.

### Changed

- The event engine is faster, with identical results: 1.1-1.9x against 0.1.0 on the
  benchmark scenarios both versions run, and 1.5-2.1x against 0.2.0 before tuning
  (`docs/guide.md`, section Performance). Market views box the calendar once, each
  bar is priced and valued once, bars and path phases on which no order can act are
  skipped, and orders are copied and validated more cheaply.
- A stop-limit order triggered intrabar with its limit beyond the stop now works as a
  limit for the rest of the bar's path; it used to wait for later bars.
- `walk_forward` runs the in-sample fits of all windows before the out-of-sample runs
  (the results are unchanged).
- `research.deflated` became a compatibility alias of `validation.sharpe`.
- The README is a shorter front page; reference material moved to `docs/`.

### Fixed

- The engine called a halted strategy's `on_fill` for the kill switch's own fills,
  so its intents could cancel the flattening orders.
- Orders for symbols outside the feed failed late with a `KeyError`, and targets for
  them were silently dropped; both now raise `OrderError` where they are placed.
- `backtest` printed a traceback when its output pipe closed early (`| head`); it now
  exits quietly with status 141.
- `MarketView.series(..., lookback=n)` masked the whole history on every call, which
  made its cost grow with the length of the backtest.
- An unpickled `DataFeed` had writeable arrays.
- The synthetic generator's documentation overstated how exact the bar highs and lows
  are (exact marginally, not jointly).

## [0.1.0] - 2026-09-25

### Added

- Event-driven engine with a priority event queue, a read-only market view that ends
  at the current bar, and next-bar execution.
- Simulated broker: market, limit, stop and stop-limit orders, DAY and GTC, gap
  handling, fixed-bps and square-root-impact slippage, per-share and bps commissions,
  a participation cap.
- Portfolio accounting with two identities checked on every bar; risk limits and a
  drawdown kill switch; position sizing helpers.
- Metrics, round-trip trades, buy-and-hold benchmark, text report and plots.
- Grid search, walk-forward optimisation with a stitched out-of-sample curve, the
  deflated Sharpe ratio, and a vectorized fast path with a parity test.
- Strategies: SMA crossover, time-series and cross-sectional momentum, Bollinger mean
  reversion; a seeded regime-switching synthetic market; the `backtest` CLI.
