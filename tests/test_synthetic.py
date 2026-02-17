from __future__ import annotations

import numpy as np
import pytest

from backtester import ConfigError, DataFeed, generate_market, generate_multi_asset, generate_ohlcv
from backtester.data import SyntheticConfig


def test_seeded_generation_is_deterministic() -> None:
    a = generate_ohlcv(3, 1, seed=5)
    b = generate_ohlcv(3, 1, seed=5)
    c = generate_ohlcv(3, 1, seed=6)
    for sym in a:
        assert a[sym].equals(b[sym])
    assert not a["SYN01"].equals(c["SYN01"])


def test_ohlc_is_consistent_and_feed_accepts_it() -> None:
    frames = generate_ohlcv(6, 4, seed=1)
    for df in frames.values():
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df["low"] > 0).all()
        assert (df["volume"] > 0).all()
    feed = DataFeed(frames)
    assert len(feed) == 4 * 252
    assert feed.symbols == tuple(f"SYN{i:02d}" for i in range(1, 7))


def test_missing_bars_are_dropped_at_the_requested_rate() -> None:
    frames = generate_ohlcv(5, 8, seed=3, missing_prob=0.05)
    n = 8 * 252
    dropped = 1 - np.mean([len(df) / n for df in frames.values()])
    assert 0.03 < dropped < 0.07
    assert all(df.index[0] == frames["SYN01"].index[0] for df in frames.values())


def test_regimes_persist_and_bear_regime_is_more_volatile() -> None:
    market = generate_market(SyntheticConfig(n_symbols=4, n_bars=5000, seed=9))
    regime = market.market_regime.to_numpy()
    switches = np.count_nonzero(np.diff(regime))
    assert 5000 / max(switches, 1) > 40  # regimes last months, not days
    close = DataFeed(market.frames).frame("close")
    rets = np.log(close).diff().mean(axis=1).to_numpy()[1:]
    bull, bear = rets[regime[1:] == 0], rets[regime[1:] == 1]
    assert bear.std() > 1.4 * bull.std()
    assert bull.mean() > bear.mean()


def test_gaps_exist_between_close_and_next_open() -> None:
    df = generate_ohlcv(1, 5, seed=2)["SYN01"]
    gaps = np.abs(np.log(df["open"] / df["close"].shift())).dropna()
    assert gaps.max() > 0.03  # overnight jumps produce opening gaps


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_symbols": 0},
        {"transition": ((0.5, 0.4), (0.1, 0.9))},
        {"missing_prob": 1.0},
        {"overnight_variance_share": 0.0},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        SyntheticConfig(**kwargs)  # type: ignore[arg-type]


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown"):
        generate_ohlcv(2, 1, seed=1, not_a_field=3)


def test_multi_asset_universe_has_block_correlation() -> None:
    frames = generate_multi_asset(years=4, seed=3)
    assert list(frames) == [f"{p}{i:02d}" for p, n in (("EQ", 6), ("BD", 4), ("CM", 4))
                            for i in range(1, n + 1)]  # fmt: skip
    feed = DataFeed(frames)
    r = np.diff(np.log(feed.last_close), axis=0)
    corr = np.corrcoef(r.T)
    within = [corr[:6, :6], corr[6:10, 6:10], corr[10:, 10:]]
    assert min(c[np.triu_indices(len(c), 1)].mean() for c in within) > 0.4
    assert abs(corr[:6, 6:].mean()) < 0.1  # classes are driven by independent factors
    vol = r.std(axis=0) * np.sqrt(252)
    assert vol[6:10].max() < 0.5 * vol[:6].min()  # bonds are much calmer
    again = generate_multi_asset(years=4, seed=3)
    assert all(frames[s].equals(again[s]) for s in frames)
    assert not frames["EQ01"].equals(generate_multi_asset(years=4, seed=4)["EQ01"])
    small = generate_multi_asset(years=1, seed=1, n_per_class=(2, 0, 1), missing_prob=0.05)
    assert sorted(small) == ["CM01", "EQ01", "EQ02"]
    with pytest.raises(ConfigError, match="at least one"):
        generate_multi_asset(n_per_class=(0, 0, 0))
