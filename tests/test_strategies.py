from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtester import ConfigError, DataFeed
from backtester.config import BacktestConfig
from backtester.strategies import (
    BollingerMeanReversion,
    CrossSectionalMomentum,
    SmaCrossover,
    TimeSeriesMomentum,
)
from tests.conftest import bars


def trend_feed(n: int = 300) -> DataFeed:
    """Three assets: steady up-trend, flat, steady down-trend."""
    t = np.arange(n)
    rng = np.random.default_rng(0)
    frames = {}
    for sym, drift, noise in (("UP", 0.003, 0.01), ("FLAT", 0.0, 0.0), ("DOWN", -0.003, 0.01)):
        close = 100 * np.exp(drift * t + np.cumsum(rng.normal(0, noise, n)))
        rows = [(c, c * 1.002, c * 0.998, c, 1e6) for c in close]
        frames[sym] = bars(rows)
    return DataFeed(frames)


def test_sma_crossover_goes_long_the_uptrend_only() -> None:
    feed = trend_feed()
    w = SmaCrossover(fast=10, slow=50).desired_weights(feed.last_close)
    assert w is not None
    np.testing.assert_allclose(w, [1 / 3, 0, 0], atol=1e-12)
    w_ls = SmaCrossover(fast=10, slow=50, allow_short=True).desired_weights(feed.last_close)
    assert w_ls is not None
    assert w_ls[0] > 0 > w_ls[2]
    assert SmaCrossover(fast=10, slow=50).desired_weights(feed.last_close[:20]) is None


def test_tsmom_is_sign_times_inverse_volatility() -> None:
    feed = trend_feed()
    s = TimeSeriesMomentum(lookback=100, vol_lookback=50, target_vol=0.1, max_gross=10)
    closes = feed.last_close
    w = s.desired_weights(closes)
    assert w is not None
    vol = np.diff(np.log(closes[-51:]), axis=0).std(axis=0, ddof=1) * math.sqrt(252)
    np.testing.assert_allclose(w[[0, 2]], np.array([1, -1]) * 0.1 / vol[[0, 2]] / 3)
    capped = TimeSeriesMomentum(lookback=100, vol_lookback=50, target_vol=0.1, max_gross=0.2)
    wc = capped.desired_weights(closes)
    assert wc is not None
    assert np.abs(wc).sum() == pytest.approx(0.2)
    long_only = TimeSeriesMomentum(lookback=100, vol_lookback=50, allow_short=False)
    wl = long_only.desired_weights(closes)
    assert wl is not None
    assert wl[2] == 0


def test_xsmom_holds_the_top_k() -> None:
    feed = trend_feed()
    s = CrossSectionalMomentum(lookback=60, skip=5, top_k=1)
    w = s.desired_weights(feed.last_close)
    assert w is not None
    np.testing.assert_allclose(w, [1, 0, 0])
    ls = CrossSectionalMomentum(lookback=60, skip=5, top_k=1, long_short=True)
    wls = ls.desired_weights(feed.last_close)
    assert wls is not None
    np.testing.assert_allclose(wls, [1, 0, -1])
    ties = s._rank_weights(np.array([[0.1, 0.1, np.nan]]))
    np.testing.assert_allclose(ties, [[1, 0, 0]])  # ties by symbol order, NaN never chosen


def test_bollinger_state_machine() -> None:
    closes = np.r_[np.full(20, 100.0), 90.0, 95.0, 99.0, 101.0, 112.0, 99.0]
    s = BollingerMeanReversion(window=20, n_std=2.0)
    states = []
    for i in range(1, len(closes) + 1):
        w = s.desired_weights(closes[:i, None])
        states.append(None if w is None else float(w[0]))
    # enter below the band, hold until back at the mean, then short the spike
    assert states[19:] == [0.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0]
    frame = s.desired_weights_frame(pd.DataFrame({"X": closes}))
    np.testing.assert_allclose(frame["X"].to_numpy()[19:], states[19:])


def test_monthly_rebalance_only_on_first_bar_of_month(synthetic_feed: DataFeed) -> None:
    result = BacktestConfig().run(synthetic_feed, TimeSeriesMomentum(lookback=63))
    decisions = sorted({o.created_at for o in result.orders})
    index = synthetic_feed.index
    for ts in decisions[1:]:  # the first decision is the first bar with enough history
        pos = index.get_loc(ts)
        assert (index[pos].year, index[pos].month) != (index[pos - 1].year, index[pos - 1].month)
    assert len(decisions) > 20


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SmaCrossover(fast=50, slow=20),
        lambda: TimeSeriesMomentum(lookback=1),
        lambda: TimeSeriesMomentum(target_vol=0),
        lambda: CrossSectionalMomentum(lookback=10, skip=10),
        lambda: CrossSectionalMomentum(top_k=0),
        lambda: BollingerMeanReversion(window=1),
        lambda: SmaCrossover(rebalance="weekly"),  # type: ignore[arg-type]
    ],
)
def test_invalid_parameters(factory) -> None:
    with pytest.raises(ConfigError):
        factory()


def test_params_are_reported() -> None:
    assert SmaCrossover(fast=5, slow=10).params() == {
        "rebalance": "change",
        "fast": 5,
        "slow": 10,
        "allow_short": False,
    }
