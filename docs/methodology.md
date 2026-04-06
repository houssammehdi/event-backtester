# Methodology

How the engine turns decisions into fills, which biases a backtest is exposed to and
what this library does about each, and what the validation statistics measure. Every
number on this page comes from a command shown next to it, run on the synthetic data
of this repository; none of it is evidence about real markets.

- [1. Execution model](#1-execution-model)
- [2. Bias taxonomy](#2-bias-taxonomy)
- [3. Validation statistics](#3-validation-statistics)
- [4. Portfolio construction](#4-portfolio-construction)
- [5. Calibration and findings](#5-calibration-and-findings)
- [References](#references)

## 1. Execution model

### 1.1 The bar loop

Data is a set of OHLCV bars on a shared calendar. For every bar `t` the engine:

1. matches the orders working from earlier bars against bar `t` and books the fills;
2. marks the portfolio to the close of `t`, records equity, charges short borrow fees
   and checks the drawdown kill switch;
3. calls the strategy's `on_bar(view, ctx)`. The view is pinned to `t`: its arrays are
   read-only NumPy views that physically end at `t`, and asking for a later timestamp
   raises `LookAheadError`;
4. passes the strategy's intents (targets, orders, cancels) through the risk manager,
   which clips, sizes or rejects them, and hands the approved orders to the broker.

An order decided at the close of `t` can first execute on bar `t + 1`. The ordering
inside a bar is enforced by the priorities of one event queue - fills, then the
market event, then intents, then orders - so it also holds for events created while
the queue is being drained (kill-switch flattening, `on_fill` hooks, cancel-and-replace).

### 1.2 Order types and fill prices

| Order | Executes on bar `t + 1` (buy side; sell side mirrored) |
|---|---|
| Market, market-on-open | At the open (the opening auction) |
| Market-on-close | At the close (the closing auction). Decided at the close of `t`, it fills at the close of `t + 1`: the close of `t` has already printed when the decision is made, so filling there would be look-ahead |
| Limit | At the open if the open is at or below the limit (a gap: the better price). Otherwise at the limit, when the price path reaches it. Touching counts; `strict_limits=True` requires trading through the limit intrabar (at the open, touching still counts - the auction prints there) |
| Stop | At the open if the open is at or above the stop (a gap: the worse price). Otherwise at the stop, where the path crosses it |
| Stop-limit | Triggers like a stop, then works as a limit for the rest of the bar's path; if the path does not come back to the limit, it keeps working on later bars (GTC) or expires (DAY) |
| Trailing stop | A stop whose level trails the best price since the order became active - the high for a sell, the low for a buy - by `trail_amount` or `trail_percent`. The level ratchets along the intrabar path. A stand-alone trailing stop starts from the close of the decision bar; the trailing exit of a bracket starts from its entry's fill price |

Slippage moves the reference price against the order; a limit-type execution is then
capped at its limit. `DAY` orders work on the first bar after submission only, `GTC`
orders until filled or cancelled. A symbol without a bar at `t + 1` (holiday, halt,
not yet listed) cannot trade on it and is valued at its last close.

### 1.3 The intrabar path

A daily bar records four prices but not their order. The broker executes orders along
an explicit path through them, one of

- `high_first`: open -> high -> low -> close, or
- `low_first`: open -> low -> high -> close,

in three phases: the opening auction (market and market-on-open orders, and every
order the open already satisfies), the three monotone segments of the path (limits,
stops, stop-limits and trailing stops in the order the path reaches them; ties by
order id), and the closing auction (market-on-close orders).

For independent limit and stop orders the choice of path does not matter: both paths
visit every price between the low and the high, and a property test checks that the
path model reproduces the per-order rules of version 0.1 exactly for such orders. It
matters when orders interact or depend on the order of prices - both exits of a
bracket inside one bar, OCO groups, trailing stops, stop-limits whose limit is
revisited or not, and orders competing for a participation cap. For those symbols the
`intrabar` setting of the broker decides:

| `intrabar` | Path used on an ambiguous bar |
|---|---|
| `"worst"` (default) | Simulate both paths, keep the one whose fills leave the lower equity at the bar's close: the value `sum(sign * qty * (close - price)) - commission` of the bar's fills. When the stop and the target of a bracket both trade inside one bar, this is the stop |
| `"best"` | The higher of the two. For sensitivity analysis only |
| `"high_first"`, `"low_first"` | Always that path |

`worst` and `best` bound what any path assumption could give; the distance between
them is the part of a result that daily bars cannot resolve. Section 5.4 measures it
for a tight bracket strategy.

### 1.4 Linked orders: brackets and OCO

`ctx.bracket(symbol, qty, entry_type, stop_loss=..., take_profit=..., trail_amount=...)`
creates an entry and up to two exits linked to it:

- the exits are armed by the entry's first fill and work for the quantity the entry
  has opened and they have not closed yet - they can fill on the entry's own bar,
  after the entry on the path;
- the first exit fill lowers the other exit by the same amount, and ends the unfilled
  part of the entry;
- once the entry has stopped working and the open quantity is zero, the bracket is
  settled: an exit that closed part of the position is marked filled for what it
  executed, the others are cancelled; an entry that never fills (a DAY order that
  expires) cancels its exits;
- the risk manager sizes and limits the entry like any order; the exits are not sized
  (they only ever close what the entry opened) and are rejected with a rejected entry.

`ctx.oco_group()` returns a group id for `ctx.order(..., oco_group=g)`: the first fill
of any member cancels the others. All members of a group must be on one symbol.

### 1.5 Costs, capacity and targets

- **Slippage:** fixed basis points, or square-root impact
  `half-spread + eta * sigma * sqrt(qty / volume)`, with `sigma` given or estimated from
  the bar's range (Parkinson, 1980), capped at `max_impact`.
- **Commission:** basis points of notional, or per share with a minimum and a cap as a
  fraction of notional.
- **Capacity:** `max_participation` caps the quantity filled per symbol and bar at a
  share of the bar's volume, allocated in path order; the rest carries (GTC) or
  expires (DAY).
- **Targets:** `ctx.target_weights(...)` is sized with the equity and the close of the
  decision bar and executes as market orders at the next open, or as market-on-close
  orders at the next close with `target_order_type=OrderType.MARKET_ON_CLOSE`.
- **Financing:** negative cash is allowed (leverage is limited by `RiskLimits`); an
  optional annual borrow fee is charged every bar on the value of short positions.
- **Kill switch:** when the drawdown from the equity peak reaches `max_drawdown`, every
  order is cancelled, the book is flattened with GTC market orders at the next open,
  and the strategy is never called again (neither `on_bar` nor `on_fill`).

`check_invariants=True` verifies on every bar that equity equals cash plus the value of
the positions (priced independently from the feed) and that it equals initial capital
plus realised and unrealised PnL minus costs.

### 1.6 What changed in 0.2

Independent market, limit and stop orders fill as in 0.1: a property test checks the
path model against the per-order rules of 0.1 on random bars and orders, and the two
benchmark scenarios that 0.1 can run end at the same equity in both versions. One
rule changed: a stop-limit triggered intrabar with its limit on the far side of the
stop used to wait for later bars; it now works as a limit for the rest of the bar's
path and can fill on the same bar if the path returns to the limit. The new order
types, the path policy and the linked orders are additions.

## 2. Bias taxonomy

| Bias | How it gets into a backtest | What this library does | What remains your job |
|---|---|---|---|
| Look-ahead | Filling at the price that produced the signal; features, universes or normalisations computed on the full sample; label windows overlapping the test period | Views that end at the decision bar; execution from `t + 1` only (MOC at the next close); walk-forward fits on physically truncated data; purged and embargoed CV; tests that truncating the data leaves earlier results unchanged, and that the vectorized path of every strategy matches the event engine (a deliberately leaky signal fails that test) | Data that is itself not point-in-time: restated fundamentals, back-adjusted series, precomputed signals you pass in |
| Survivorship | The universe is chosen with hindsight: today's constituents, delisted names dropped (Brown et al., 1992) | Symbols can enter and leave the calendar (missing bars are not tradable and are valued at the last close) | Supplying a point-in-time universe with delisted securities and their delisting returns |
| Selection bias, data snooping | Many ideas, parameters or variants are tried on one history and the best is reported (Lo & MacKinlay, 1990; White, 2000; Harvey & Liu, 2015) | Deflated Sharpe ratio for the number of trials; White's Reality Check and Hansen's SPA over the whole family; `backtest validate`; walk-forward reports print a multiple-testing warning | Counting honestly: the tests only know the configurations you give them, not the ideas you discarded before writing the grid |
| Overfitting | Parameters fitted to noise: the in-sample optimum degrades out of sample | Probability of backtest overfitting (CSCV) with degradation slope and probability of loss; walk-forward with a stitched out-of-sample curve and the in-sample PBO of every window; purged K-fold and CPCV backtest paths | Choosing a search that is small relative to the data, and reading a high PBO as a verdict on the search |
| Transaction-cost optimism | Zero or constant costs, unlimited liquidity, fills at the most favourable intrabar price (Novy-Marx & Velikov, 2016) | Next-bar execution; bps or square-root-impact slippage; commissions; a participation cap; gaps fill stops at the worse open; worst-case intrabar path by default; strict limits; borrow fees | Calibrating costs to the instruments traded; queue position, spreads that widen in stress, short availability and multi-day impact are not modelled |
| Sampling luck | One history is one draw; a Sharpe ratio estimated from ten years is uncertain | Stationary-bootstrap intervals, PSR, minimum track record length; the seed study in the README runs every strategy on 20 synthetic markets | Deciding what uncertainty is acceptable before looking at the result |

The first and the last rows are about information; the middle rows are about choices.
Look-ahead can be ruled out mechanically, and the engine does. Selection and
overfitting cannot be ruled out, only measured, and the measurements depend on the
researcher reporting everything that was tried.

## 3. Validation statistics

`backtester.validation` implements the statistics below; `backtest validate` and the
tear sheet (`backtest run ... --report`) put them in reports. Sharpe ratios in the
formulas are per period; `T` is the number of returns, `g3` their skewness and `g4`
their (non-excess) kurtosis.

**Sharpe-ratio inference** (Lo, 2002; Mertens, 2002; Bailey & López de Prado, 2012,
2014). For iid but non-normal returns the estimated Sharpe ratio has standard error
`sqrt((1 - g3 SR + (g4 - 1) / 4 SR^2) / (T - 1))`.

- The *probabilistic Sharpe ratio* `PSR(SR*)` is the probability that the true Sharpe
  ratio exceeds a benchmark `SR*`: the normal CDF of `(SR - SR*)` over that standard
  error.
- The *minimum track record length* is the `T` at which `PSR(SR*)` reaches a given
  confidence: `1 + (1 - g3 SR + (g4 - 1) / 4 SR^2) (z / (SR - SR*))^2`.
- The *deflated Sharpe ratio* is the PSR against the Sharpe ratio expected from the
  best of `N` skill-less trials, `sqrt(V) ((1 - gamma) Z(1 - 1/N) + gamma Z(1 - 1/(N e)))`,
  with `V` the variance of the trials' Sharpe ratios and `gamma` the Euler-Mascheroni
  constant. Correlated trials overstate the effective `N`, which makes it
  conservative.

**Stationary bootstrap** (Politis & Romano, 1994). Resamples blocks of consecutive
returns with geometric lengths, so serial dependence survives the resampling. The mean
block length is chosen automatically (Politis & White, 2004, with the correction of
Patton, Politis & White, 2009). Intervals are percentile (default) or basic intervals
for the Sharpe ratio, CAGR, maximum drawdown, volatility, mean and any user statistic.

**Probability of backtest overfitting** (Bailey, Borwein, López de Prado & Zhu, 2017).
The performance matrix of a search (one column of returns per configuration) is cut
into `S` blocks; for every one of the `C(S, S/2)` ways to take half of them as the
in-sample set, the in-sample best configuration is ranked among all configurations on
the other half. The PBO is the share of splits in which it lands below the median. It
comes with the logit distribution, the slope of out-of-sample on in-sample
performance (degradation) and the probability of an out-of-sample loss.

**Reality Check and SPA** (White, 2000; Hansen, 2005). Test `H0: no strategy of the
family beats the benchmark in expectation`, with the stationary bootstrap resampling
the same dates for every strategy. The Reality Check uses the largest mean
out-performance; the SPA studentises it and recentres only strategies that are not
clearly worse than the benchmark, which keeps its power when poor strategies are added
to the family. The *consistent* p-value is the one to report; the *lower* and *upper*
p-values bracket it.

**Purged cross-validation** (López de Prado, 2018, ch. 7 and 12). An observation that
depends on an interval of time leaks into any test block that interval overlaps.
Purged K-fold removes such observations from the training set, and an embargo also
drops the observations just after each test block. Combinatorial purged CV uses every
choice of `k` of `N` groups as the test set, so that the out-of-sample predictions can
be stitched into `C(N - 1, k - 1)` complete backtest paths instead of one;
`cpcv_backtest` runs a parameter selection inside each split and reports the Sharpe
ratio of every path.

## 4. Portfolio construction

`backtester.portfolio` provides the allocation methods used by `AllocationStrategy`:

- **Covariance:** the sample covariance, and Ledoit-Wolf shrinkage towards a
  constant-correlation target (Ledoit & Wolf, 2004a) or towards a scaled identity
  (Ledoit & Wolf, 2004b). Both are tested against a direct transcription of the
  published estimators.
- **Inverse volatility**, **minimum variance** and **mean-variance**, long-only with an
  optional weight cap, solved exactly by an active-set method on the capped simplex
  (the tests check the KKT conditions and compare with brute-force enumeration of the
  active sets).
- **Equal risk contribution** (risk parity) by Newton's method on Spinu's (2013)
  convex formulation; the tests check that every asset contributes the same risk.
- **Hierarchical risk parity** (López de Prado, 2016): correlation distance, single
  linkage, quasi-diagonalisation and recursive bisection, tested on a four-asset
  example worked by hand. As published, the weights depend on how the assets are
  labelled: a merge of two single assets lists the smaller label first, and the
  bisection cuts the leaf order by position rather than along the tree, so
  relabelling can move an asset to the other half.
  `test_hrp_depends_on_the_labels_of_the_assets` shows a six-asset case where some
  weights change by more than 10 %.

## 5. Calibration and findings

The numbers in this section are printed by `python scripts/validation_calibration.py`
(5.1-5.3; about 4.5 minutes on the shared 4-vCPU cloud VM used here) and by the
examples named in 5.4-5.5. Everything is seeded, so the numbers do not depend on the
machine; the run times do.

### 5.1 Size and power of the data-snooping tests

Under the null every strategy is pure noise with zero mean, so a test at level 5 %
should reject in 5 % of the simulations:

```text
Size of the data-snooping tests: rejection rate under the null
(500 observations, iid, 500 bootstrap resamples; 1000 and 4000 simulations)
test                           at 5%  at 10%
-----------------------------  -----  ------
RC, 10 strategies               5.0%    9.7%
SPA lower, 10 strategies        7.5%   15.8%
SPA consistent, 10 strategies   5.0%   10.3%
SPA upper, 10 strategies        4.9%    9.9%
RC, 1 strategy                  5.6%   10.2%
SPA lower, 1 strategy           5.6%   10.2%
SPA consistent, 1 strategy      5.6%   10.2%
SPA upper, 1 strategy           5.6%   10.2%
```

The Reality Check and the consistent and upper SPA p-values hold their level. The
lower p-value over-rejects (7.5 % at 5 %), as its construction implies - it recentres
only strategies with a positive sample mean and is a bound, not the p-value to
report. With a single strategy the variants coincide.

The reason to prefer the SPA shows when poor strategies join the family. One strategy
has a per-period Sharpe ratio of 0.12 and the other `k` lose money:

```text
Power at 5%: one strategy with per-period Sharpe 0.12, plus k losing strategies
(500 observations, 300 simulations each)
k   Reality Check  SPA consistent
--  -------------  --------------
0             86%             86%
5             66%             87%
20            43%             83%
50            30%             83%
```

Every useless addition costs the Reality Check power (86 % to 30 % with 50 of them);
the SPA, which studentises and does not recentre clearly inferior strategies, keeps
it (Hansen, 2005). A note for anyone comparing with other implementations: the SPA of
the `arch` package (version 8.0.0, used as a cross-check during development)
compares the largest raw mean differential with its bootstrap maxima, and uses the
variances only in the consistent recentring rule, so its statistic is not
studentised and its p-values are not comparable with this one's.

### 5.2 Coverage of bootstrap intervals

```text
Coverage of stationary-bootstrap percentile intervals, iid normal returns
(500 observations, 500 resamples, 1000 samples)
nominal   mean  Sharpe  volatility
-------  -----  ------  ----------
90%      89.8%   90.0%       88.6%
95%      93.6%   93.4%       94.6%
```

Percentile intervals from the stationary bootstrap cover the true mean, Sharpe ratio
and volatility close to their nominal level on iid data of the length of two years of
daily returns. At 95 % the coverage of the mean and the Sharpe ratio falls about 1.5
points short, a known small-sample property of percentile intervals.

### 5.3 The probability of backtest overfitting

```text
PBO of 20 pure-noise strategies, mean over 200 families: 0.508
PBO with one skilled strategy among 19 noise strategies (40 seeds, 10 blocks)
per-period Sharpe  observations  mean PBO  max PBO
-----------------  ------------  --------  -------
0.15                       1000     0.061    0.329
0.15                       2000     0.005    0.067
0.2                        1000     0.005    0.083
0.2                        2000     0.000    0.004
```

Pure noise gives a PBO of one half, as it should. A strategy with real skill is
recognised when the sample is long enough: with a per-period Sharpe ratio of 0.15
(about 2.4 annualised for daily data) and 1,000 observations the PBO averages 6 %
but reached 33 % in one of the 40 families, and doubling the sample brings the worst
case down to 7 %. A low PBO is evidence; a single moderate one is not a verdict.

### 5.4 Intrabar sensitivity

`python examples/intrabar_sensitivity.py` runs a breakout strategy with brackets
(buy-stop entry at the 20-day high, stop-loss and target at multiples of the average
range) under the four intrabar policies:

```text
Bracket breakout: 20-day high entry, stop 2.0 ATR, target 3.0 ATR (5 symbols, 10 years, 2 + 1 bps costs)
intrabar    trades  hit rate    CAGR  Sharpe  max DD  ambiguous exits
----------  ------  --------  ------  ------  ------  ---------------
worst          438     40.4%  -1.43%   -0.06  39.92%                0
low_first      437     40.5%  -1.34%   -0.05  39.93%                0
high_first     438     40.4%  -1.43%   -0.06  39.92%                0
best           437     40.5%  -1.34%   -0.05  39.93%                0

Bracket breakout: 20-day high entry, stop 0.5 ATR, target 0.75 ATR (5 symbols, 10 years, 2 + 1 bps costs)
intrabar    trades  hit rate     CAGR  Sharpe  max DD  ambiguous exits
----------  ------  --------  -------  ------  ------  ---------------
worst         1600     28.6%  -16.80%   -4.02  84.10%              106
low_first     1237     43.5%   -3.03%   -0.61  28.44%              104
high_first    1600     29.9%  -15.66%   -3.80  81.78%              106
best          1237     46.0%   -1.08%   -0.20  20.61%              104
```

With a wide bracket no exit falls on a bar that reaches both the stop and the target,
and the four policies agree to within 0.1 % a year. With a tight bracket about 105
exits are ambiguous, and the result ranges from -16.8 % a year (worst case) to -1.1 %
(best case). The ambiguous bars are a small share of the exits, but each one decides
between a win and a loss and changes what the strategy does next: the two bounds
differ by a third in the number of trades. Any backtest of tight stops on daily bars
carries this spread; reporting the worst case is the conservative choice, and
reporting only one fixed path hides it.

### 5.5 Portfolio construction out of sample

`python examples/allocation_study.py` compares equal weight, inverse volatility,
minimum variance, ERC and HRP on the 14-asset synthetic universe
(`generate_multi_asset`), rebalanced monthly with a 252-bar Ledoit-Wolf covariance,
2 bps slippage, 1 bps commission and a 10 % participation cap, and then validates the
comparison itself:

```text
Universe: 14 assets, 2520 bars, seed 42

method   CAGR     vol     Sharpe [95% CI]       max DD [95% CI]  turnover  costs
------  -----  ------  ------------------  --------------------  --------  -----
ew      0.33%  12.77%  0.09 [-0.51, 0.71]  29.0% [18.9%, 61.2%]       62%  1,733
iv      1.42%   6.79%  0.24 [-0.37, 0.87]   15.7% [9.5%, 34.8%]       46%  1,381
minvar  1.50%   4.38%  0.36 [-0.28, 0.99]   12.6% [5.7%, 21.8%]      102%  3,064
erc     1.60%   6.68%  0.27 [-0.34, 0.91]   16.1% [9.3%, 33.4%]       51%  1,510
hrp     1.76%   4.87%  0.38 [-0.22, 1.03]   13.3% [6.2%, 23.4%]      152%  4,574

Selection among the five methods (H0: none beats EW)
----------------------------------------------------  -----
Best full-sample Sharpe                                 hrp
PBO of picking the best method                        68.9%
SPA p-value, raw returns                              0.410
SPA p-value, at EW's volatility                       0.150
Reality Check p-value, at EW's volatility             0.220
At EW's volatility, hrp minus EW earns 3.73% a year [95% CI -4.90%, 12.15%]

Methods chosen per window: minvar, erc, erc, ew, hrp, hrp
Walk-forward OOS    Sharpe         95% CI   CAGR  max DD
------------------  ------  -------------  -----  ------
selected in-sample    0.27  [-0.59, 1.07]  1.85%  20.56%
always EW             0.23  [-0.59, 1.03]  2.22%  24.68%

Sharpe ratios over 20 seeds (10 years each, same costs)
method  mean   std    min   max  beats EW
------  ----  ----  -----  ----  --------
ew      0.13  0.46  -0.51  0.99
iv      0.29  0.44  -0.42  1.08      100%
minvar  0.41  0.36  -0.30  0.90       80%
erc     0.29  0.46  -0.45  1.14       95%
hrp     0.42  0.34  -0.29  0.98       85%
SPA at EW's volatility rejects 'nothing beats EW' at 5% in 5 of 20 markets.
```

The risk-based methods beat equal weight on most of the 20 markets (inverse volatility
on all of them): equal weight lets the most volatile asset classes dominate the
portfolio's risk. On the headline market, though, the in-sample ranking of the five
methods does not carry over: the PBO of picking the best one is 69 %, above the one
half of pure noise, because close competitors swap places from one half of the sample
to the other. No method beats equal weight at the 5 % level, on raw returns (SPA p = 0.41) or
at equal volatility (p = 0.15), and selecting the method by walk-forward gave about
the out-of-sample Sharpe ratio of always holding equal weight (0.27 against 0.23,
with intervals that overlap almost entirely). The honest summary: risk-based
weighting helps on average on this generator, and which risk-based method is best is
not identifiable from ten years of data.

## References

- Bailey, D. H., Borwein, J. M., López de Prado, M. and Zhu, Q. J. (2017). The
  probability of backtest overfitting. *Journal of Computational Finance* 20(4), 39-69.
- Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier.
  *Journal of Risk* 15(2), 3-44.
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting
  for selection bias, backtest overfitting and non-normality. *Journal of Portfolio
  Management* 40(5), 94-107.
- Brown, S. J., Goetzmann, W., Ibbotson, R. G. and Ross, S. A. (1992). Survivorship
  bias in performance studies. *Review of Financial Studies* 5(4), 553-580.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business &
  Economic Statistics* 23(4), 365-380.
- Harvey, C. R. and Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*
  42(1), 13-28.
- Ledoit, O. and Wolf, M. (2004a). Honey, I shrunk the sample covariance matrix.
  *Journal of Portfolio Management* 30(4), 110-119.
- Ledoit, O. and Wolf, M. (2004b). A well-conditioned estimator for large-dimensional
  covariance matrices. *Journal of Multivariate Analysis* 88(2), 365-411.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*
  58(4), 36-52.
- Lo, A. W. and MacKinlay, A. C. (1990). Data-snooping biases in tests of financial
  asset pricing models. *Review of Financial Studies* 3(3), 431-467.
- López de Prado, M. (2016). Building diversified portfolios that outperform out of
  sample. *Journal of Portfolio Management* 42(4), 59-69.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002). Working
  paper, University of Basel.
- Novy-Marx, R. and Velikov, M. (2016). A taxonomy of anomalies and their trading
  costs. *Review of Financial Studies* 29(1), 104-147.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the
  rate of return. *Journal of Business* 53(1), 61-65.
- Patton, A., Politis, D. N. and White, H. (2009). Correction to "Automatic
  block-length selection for the dependent bootstrap" by D. Politis and H. White.
  *Econometric Reviews* 28(4), 372-375.
- Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *Journal of the
  American Statistical Association* 89(428), 1303-1313.
- Politis, D. N. and White, H. (2004). Automatic block-length selection for the
  dependent bootstrap. *Econometric Reviews* 23(1), 53-70.
- Spinu, F. (2013). An algorithm for computing risk parity weights. Working paper,
  SSRN 2297383.
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5),
  1097-1126.
