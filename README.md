# event-backtester

[![CI](https://github.com/houssammehdi/event-backtester/actions/workflows/ci.yml/badge.svg)](https://github.com/houssammehdi/event-backtester/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

**Event-driven backtesting and validation for systematic trading strategies.**
Strategies see a read-only view of the market that ends at the current bar, so
look-ahead is ruled out by construction. Orders fill on later bars along an explicit
intrabar path, with costs, a participation cap and the worst case when a daily bar
cannot say what came first. And because a backtest is one draw of history - usually
the best of many tried - the library measures how much of a result could be luck:
the probability of backtest overfitting, White's Reality Check and Hansen's SPA,
stationary-bootstrap intervals, probabilistic, deflated and minimum-track-record
Sharpe statistics, and purged combinatorial cross-validation. Runtime dependencies
are NumPy and pandas; everything runs offline on seeded synthetic markets.

> **Disclaimer.** Research software, not investment advice. Results on synthetic data
> say nothing about how a strategy would do in real markets.

![Tear sheet written by `backtest run ... --report`](docs/tear_sheet.png)

## What is in it

- **Engine.** One priority event queue (fills, market event, intents, orders), next-bar
  execution, portfolio accounting checked against two identities, risk limits and a
  drawdown kill switch.
- **Orders.** Market, limit, stop, stop-limit, trailing stops, market-on-open and
  market-on-close; brackets (entry with stop-loss and take-profit) and OCO groups;
  gaps, slippage, commissions and volume caps. When both exits of a bracket trade in
  one bar, the worse outcome (the stop) is assumed by default.
- **Validation.** PBO via CSCV, Reality Check and SPA, the stationary bootstrap with
  automatic block length, PSR, DSR and MinTRL, purged K-fold and CPCV with backtest
  paths - calibrated on simulations with a known answer (see
  [methodology](docs/methodology.md#5-calibration-and-findings)).
- **Portfolio construction.** Ledoit-Wolf covariance, inverse volatility, long-only
  minimum variance and mean-variance, equal risk contribution and hierarchical risk
  parity, as a rebalancing strategy.
- **Research.** Grid search and walk-forward optimisation (serial or in a process pool,
  with identical results), a vectorized fast path that matches the event engine to
  floating-point precision, and a self-contained HTML tear sheet.

## Quickstart

```bash
git clone https://github.com/houssammehdi/event-backtester.git
cd event-backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,plot]"

backtest run --strategy tsmom --symbols 5 --years 10 --seed 42 --report tsmom.html
backtest validate --strategy tsmom --symbols 5 --years 10 --seed 42
backtest walkforward --strategy tsmom --symbols 5 --years 10 --seed 42 --jobs 4
backtest strategies
```

```python
from backtester import BpsCommission, DataFeed, FixedBpsSlippage, generate_ohlcv
from backtester.config import BacktestConfig
from backtester.strategies import TimeSeriesMomentum

feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))  # or DataFeed.from_csv(path)
config = BacktestConfig(slippage=FixedBpsSlippage(2), commission=BpsCommission(1))
result = config.run(feed, TimeSeriesMomentum())
print(result.report())
result.tear_sheet("tsmom.html")
```

## Is the result real?

`backtest validate` backtests every configuration of a grid and asks how much of the
winner is selection. Real output, with the default grid and costs (2 bps slippage,
1 bps commission, 10 % participation cap):

```console
$ backtest validate --strategy tsmom --symbols 5 --years 10 --seed 42
Validation: tsmom  grid: lookback=[63, 126, 252], target_vol=[0.1, 0.2]
6 configurations, each backtested on all 2520 bars (2014-01-02 -> 2023-08-30, 5 symbols)

Configuration (best first)    Sharpe    CAGR  Max DD
----------------------------  ------  ------  ------
lookback=252, target_vol=0.1    0.95   6.11%  11.54%
lookback=252, target_vol=0.2    0.95  12.11%  22.15%
lookback=126, target_vol=0.2    0.58   7.08%  21.19%
lookback=126, target_vol=0.1    0.57   3.69%  10.83%
lookback=63, target_vol=0.2     0.45   4.97%  29.84%
lookback=63, target_vol=0.1     0.45   2.63%  16.01%

Selected: lookback=252, target_vol=0.1  (best full-sample Sharpe of 6 configurations)

Selected configuration       estimate          95% interval
--------------------------  ---------  --------------------
Sharpe ratio (annualised)        0.95          0.35 .. 1.54
CAGR                            6.11%       2.10% .. 10.07%
Max drawdown                   11.54%       6.10% .. 18.64%
Annual volatility               6.45%        6.06% .. 6.85%
PSR (true Sharpe > 0)           0.999
Min. track record (95%)     3.0 years  (sample: 10.0 years)
Deflated Sharpe (6 trials)      0.979
Stationary bootstrap: 2000 resamples, mean block length 3.4 bars; PSR and MinTRL adjust for skew -0.11 and kurtosis 7.7.

Backtest overfitting (CSCV)
------------------------------------  -----
Probability of backtest overfitting   15.6%
Degradation slope (OOS on IS Sharpe)  -0.86
P(selected loses out of sample)        1.5%
16 blocks of 157 bars, 12870 train/test combinations.

Data snooping vs cash             p-value
--------------------------  -------------
White's Reality Check               0.004
Hansen SPA (consistent)             0.004
Hansen SPA (lower / upper)  0.004 / 0.004
H0: none of the 6 configurations beats cash; 2000 resamples, block length 1.3 bars.

Combinatorial purged CV
----------------------------  ------------------
Backtest paths                                 5
OOS Sharpe: mean / min / max  0.80 / 0.52 / 0.97
6 groups, 2 per test set; training returns within 253 bars of a test block purged, embargo 0 bars.

Execution: next-bar open / intrabar, slippage=FixedBpsSlippage(bps=2.0), commission=BpsCommission(bps=1.0), max participation=0.1
Ran 6 backtests in 2.0s
```

On this market the search looks healthy: the winner beats cash beyond what six tries
would give by luck, and a PBO of 16 % says the in-sample winner usually stays in the
top half out of sample. It is still one market. The seed study runs every strategy on
20 synthetic markets (`python examples/seed_study.py`, vectorized path, same costs):

```text
Sharpe ratios across 20 seeds (5 symbols, 10 years, 2 bps + 1 bps costs)
strategy    mean Sharpe   std    min   max  beats B&H
----------  -----------  ----  -----  ----  ---------
sma                0.26  0.37  -0.42  0.80        55%
tsmom              0.24  0.28  -0.29  0.64        40%
xsmom              0.32  0.39  -0.54  1.03        65%
bollinger         -0.20  0.35  -0.82  0.60        15%
allocation         0.21  0.36  -0.51  0.65        40%
buy & hold         0.27  0.37  -0.47  1.01
```

Seed 42 is a good draw for time-series momentum: its Sharpe ratio of 0.95 is above
the best of these 20 markets (0.64) and far above their average (0.24). On this
generator the trend strategies roughly match buy-and-hold on average, and mean
reversion loses. Walk-forward testing (`backtest walkforward`, same data) shows the
shrinkage within one market too: the stitched out-of-sample Sharpe ratio is 0.65
(95 % interval -0.13 to 1.42) against an average in-sample optimum of 0.94, and its
six years out of sample are shorter than the 6.5 years needed to call it positive at
95 %.

Two more studies in [`examples/`](examples) are deliberately unflattering. Choosing
among five portfolio-construction methods on a multi-asset universe has a PBO of 69 %,
and no method beats equal weight at the 5 % level on that market
(`allocation_study.py`). A breakout strategy with a tight bracket loses 17 % a year
under the worst-case intrabar assumption and 1 % under the best case: daily bars
cannot tell which of its stop and target traded first (`intrabar_sensitivity.py`).
[Methodology, section 5](docs/methodology.md#5-calibration-and-findings) has the
numbers.

## Performance

Median wall time of `python scripts/benchmark_engine.py` (2,520 daily bars per run,
2 bps slippage, 1 bps commission, 10 % participation cap), 7 interleaved runs per
version:

| Scenario | 0.1.0 | 0.2.0 |
|---|---|---|
| Time-series momentum, 5 symbols, monthly targets | 0.73 s | 0.34 s (7,400 bars/s) |
| SMA crossover, 20 symbols, daily targets | 1.58 s | 1.33 s (1,900 bars/s) |
| Bracket breakout, 5 symbols (not available in 0.1.0) | - | 2.09 s (1,200 bars/s) |
| HRP allocation, 14 assets (not available in 0.1.0) | - | 0.41 s (6,100 bars/s) |

Measured on a shared 4-vCPU cloud VM whose load average stayed around 7.7 from other
jobs, so the times are indicative; the versions were interleaved to keep the
comparison fair. Both versions give the same final equity. A grid search of 18
configurations took 8.4 s serially and 6.1 s with `n_jobs=4` on the same loaded
machine (identical results). Details in the [guide](docs/guide.md#performance).

## Documentation

- [Methodology](docs/methodology.md): execution model, intrabar path, brackets and
  auctions; the bias taxonomy (look-ahead, survivorship, data snooping, overfitting,
  cost optimism) and which tool addresses each; the validation statistics, their
  calibration and references.
- [User guide](docs/guide.md): CLI, writing strategies, orders, portfolio
  construction, research tools, validation API, tear sheet, metrics, benchmarks,
  development.
- [Changelog](CHANGELOG.md).
- [Examples](examples): quickstart, a custom stop-order strategy, strategy comparison,
  walk-forward, vectorized parity, seed study, allocation study, intrabar
  sensitivity. All run offline.

## Limitations

- Bar data only: no ticks, quotes or queue position; the intrabar path is one of two
  canonical orders, bounded by the worst and best case.
- No corporate actions and no point-in-time universe: supply adjusted,
  survivorship-free data.
- One currency, no interest on cash, a flat borrow rate, no margin calls beyond the
  kill switch; no futures multipliers.
- Impact is per fill (square-root law), with no decay across bars and no cross-impact.
- The validation statistics only know the configurations they are given; ideas
  discarded before the grid was written are not counted.
- Synthetic data exercises the machinery and makes results reproducible; it does not
  validate a strategy.

## License

[MIT](LICENSE) © 2026 Houssam Mehdi
