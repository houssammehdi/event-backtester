"""Seeded synthetic market: regime-switching geometric Brownian motion with jumps.

The generator produces realistic-looking daily OHLCV panels so that every example and
test runs offline and reproducibly:

* A shared **market regime** follows a two-state Markov chain (calm bull / volatile
  bear). Regimes persist for months, which gives the series genuine trends - the kind
  of structure momentum strategies are designed to capture.
* Each symbol loads on the market factor with its own beta and adds an idiosyncratic
  component whose drift also switches regimes (source of cross-sectional dispersion).
* **Jumps** arrive as a Poisson process and land overnight, producing opening gaps.
* The daily log return is split into an overnight and an intraday part. The high is
  drawn exactly from the distribution of the maximum of a Brownian bridge between open
  and close, and the low from that of its minimum, so
  ``low <= min(open, close) <= max(open, close) <= high`` holds by construction. The
  two are drawn independently: each marginal is exact, their joint law is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt
import pandas as pd

from backtester.errors import ConfigError

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Regime:
    """Annualised drift and volatility of one market regime."""

    drift: float
    volatility: float


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Parameters of the synthetic market. All rates are annualised."""

    n_symbols: int = 5
    n_bars: int = 10 * TRADING_DAYS
    start: str = "2014-01-02"
    seed: int | None = 42
    regimes: tuple[Regime, ...] = (Regime(0.14, 0.13), Regime(-0.22, 0.28))
    transition: tuple[tuple[float, ...], ...] = ((0.997, 0.003), (0.010, 0.990))
    """Daily Markov transition matrix between ``regimes`` (rows sum to one)."""
    beta_range: tuple[float, float] = (0.6, 1.4)
    idio_volatility: float = 0.16
    idio_drifts: tuple[float, ...] = (0.12, -0.12)
    idio_switch_prob: float = 0.003
    jump_intensity: float = 1.5
    """Expected market jumps per year."""
    jump_mean: float = -0.02
    jump_std: float = 0.04
    idio_jump_intensity: float = 1.0
    overnight_variance_share: float = 0.25
    start_price_range: tuple[float, float] = (20.0, 200.0)
    base_volume: float = 2_000_000.0
    missing_prob: float = 0.0
    """Probability that any given (bar, symbol) is dropped to simulate missing data."""
    symbol_prefix: str = "SYN"

    def __post_init__(self) -> None:
        if self.n_symbols < 1 or self.n_bars < 2:
            raise ConfigError("need at least one symbol and two bars")
        k = len(self.regimes)
        matrix = np.asarray(self.transition, dtype=float)
        if matrix.shape != (k, k) or not np.allclose(matrix.sum(axis=1), 1.0):
            raise ConfigError("transition must be a square stochastic matrix matching regimes")
        if (matrix < 0).any():
            raise ConfigError("transition probabilities must be non-negative")
        if not 0 <= self.missing_prob < 1:
            raise ConfigError("missing_prob must be in [0, 1)")
        if not 0 < self.overnight_variance_share < 1:
            raise ConfigError("overnight_variance_share must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class SyntheticMarket:
    """Output of :func:`generate_market`."""

    frames: dict[str, pd.DataFrame]
    market_regime: pd.Series
    betas: dict[str, float] = field(default_factory=dict)


def _markov_chain(
    rng: np.random.Generator, matrix: npt.NDArray[np.float64], n: int, start: int = 0
) -> npt.NDArray[np.int64]:
    states = np.empty(n, dtype=np.int64)
    states[0] = start
    uniforms = rng.random(n)
    cumulative = np.cumsum(matrix, axis=1)
    for t in range(1, n):
        states[t] = int(np.searchsorted(cumulative[states[t - 1]], uniforms[t], side="right"))
    out: npt.NDArray[np.int64] = np.minimum(states, matrix.shape[0] - 1)
    return out


def _bridge_extremes(
    rng: np.random.Generator,
    x: npt.NDArray[np.float64],
    sigma: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sample (max, min) of Brownian bridges from 0 to ``x`` with total std ``sigma``."""
    var = sigma**2
    u = rng.random(x.shape)
    v = rng.random(x.shape)
    high = 0.5 * (x + np.sqrt(x**2 - 2.0 * var * np.log1p(-u)))
    low = 0.5 * (x - np.sqrt(x**2 - 2.0 * var * np.log1p(-v)))
    return high, low


def generate_market(config: SyntheticConfig | None = None) -> SyntheticMarket:
    """Generate a synthetic OHLCV panel according to ``config``."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n, m = cfg.n_bars, cfg.n_symbols
    dt = 1.0 / TRADING_DAYS
    w_on = cfg.overnight_variance_share

    # --- market factor -------------------------------------------------------------
    matrix = np.asarray(cfg.transition, dtype=float)
    regime = _markov_chain(rng, matrix, n)
    drift = np.array([r.drift for r in cfg.regimes])[regime]
    vol = np.array([r.volatility for r in cfg.regimes])[regime]
    mu_dt = (drift - 0.5 * vol**2) * dt
    sd = vol * np.sqrt(dt)
    mkt_on = w_on * mu_dt + np.sqrt(w_on) * sd * rng.standard_normal(n)
    mkt_id = (1 - w_on) * mu_dt + np.sqrt(1 - w_on) * sd * rng.standard_normal(n)
    n_jumps = rng.poisson(cfg.jump_intensity * dt, n)
    mkt_on += n_jumps * cfg.jump_mean + np.sqrt(n_jumps) * cfg.jump_std * rng.standard_normal(n)

    # --- idiosyncratic components ---------------------------------------------------
    betas = rng.uniform(*cfg.beta_range, m)
    k = len(cfg.idio_drifts)
    stay = 1.0 - cfg.idio_switch_prob
    idio_matrix = np.full((k, k), cfg.idio_switch_prob / max(k - 1, 1))
    np.fill_diagonal(idio_matrix, stay if k > 1 else 1.0)
    idio_state = np.column_stack(
        [_markov_chain(rng, idio_matrix, n, start=int(rng.integers(k))) for _ in range(m)]
    )
    idio_mu = np.asarray(cfg.idio_drifts, dtype=float)[idio_state]
    idio_sd = cfg.idio_volatility * np.sqrt(dt)
    idio_mu_dt = (idio_mu - 0.5 * cfg.idio_volatility**2) * dt
    shape = (n, m)
    idio_on = w_on * idio_mu_dt + np.sqrt(w_on) * idio_sd * rng.standard_normal(shape)
    idio_id = (1 - w_on) * idio_mu_dt + np.sqrt(1 - w_on) * idio_sd * rng.standard_normal(shape)
    idio_jumps = rng.poisson(cfg.idio_jump_intensity * dt, shape)
    idio_on += idio_jumps * rng.normal(0.0, cfg.jump_std * 1.5, shape)

    r_on = betas * mkt_on[:, None] + idio_on
    r_id = betas * mkt_id[:, None] + idio_id
    r_on[0] = 0.0  # the first open equals the starting price

    # --- prices ---------------------------------------------------------------------
    p0 = rng.uniform(*cfg.start_price_range, m)
    log_close = np.log(p0) + np.cumsum(r_on + r_id, axis=0)
    log_open = log_close - r_id
    intraday_sd = np.sqrt((betas * sd[:, None] * np.sqrt(1 - w_on)) ** 2 + (1 - w_on) * idio_sd**2)
    hi_ext, lo_ext = _bridge_extremes(rng, r_id, intraday_sd)
    opens = np.exp(log_open)
    closes = np.exp(log_close)
    highs = np.maximum(opens * np.exp(hi_ext), np.maximum(opens, closes))
    lows = np.minimum(opens * np.exp(lo_ext), np.minimum(opens, closes))

    base = cfg.base_volume * rng.lognormal(0.0, 0.5, m)
    abs_move = np.abs(r_on + r_id) / (np.sqrt(dt) * 0.25)
    volume = base * rng.lognormal(0.0, 0.25, shape) * (1.0 + 0.5 * abs_move)
    volume = np.round(volume)

    index = pd.bdate_range(cfg.start, periods=n, name="date")
    frames: dict[str, pd.DataFrame] = {}
    symbols = [f"{cfg.symbol_prefix}{j + 1:02d}" for j in range(m)]
    drop = rng.random(shape) < cfg.missing_prob
    drop[0] = False
    for j, sym in enumerate(symbols):
        df = pd.DataFrame(
            {
                "open": opens[:, j],
                "high": highs[:, j],
                "low": lows[:, j],
                "close": closes[:, j],
                "volume": volume[:, j],
            },
            index=index,
        )
        frames[sym] = df.loc[~drop[:, j]].copy()
    regime_series = pd.Series(regime, index=index, name="regime")
    return SyntheticMarket(
        frames=frames,
        market_regime=regime_series,
        betas={s: float(b) for s, b in zip(symbols, betas, strict=True)},
    )


def generate_ohlcv(
    n_symbols: int = 5,
    years: float = 10.0,
    seed: int | None = 42,
    **overrides: object,
) -> dict[str, pd.DataFrame]:
    """Convenience wrapper returning only the ``symbol -> OHLCV`` frames.

    ``overrides`` are forwarded to :class:`SyntheticConfig` (e.g. ``missing_prob=0.02``).
    """
    cfg = replace(
        SyntheticConfig(),
        n_symbols=n_symbols,
        n_bars=max(round(years * TRADING_DAYS), 2),
        seed=seed,
    )
    if overrides:
        valid = set(SyntheticConfig.__dataclass_fields__)
        unknown = set(overrides) - valid
        if unknown:
            raise ConfigError(f"unknown synthetic config fields: {sorted(unknown)}")
        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]
    return generate_market(cfg).frames
