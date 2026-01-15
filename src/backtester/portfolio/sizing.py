"""Position sizing helpers: fixed fractional, volatility targeting, target weights."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from backtester.errors import ConfigError


def round_to_lot(quantity: float, lot_size: float | None) -> float:
    """Round ``quantity`` toward zero to a multiple of ``lot_size``.

    ``lot_size=None`` keeps fractional quantities unchanged.
    """
    if lot_size is None:
        return quantity
    if lot_size <= 0:
        raise ConfigError("lot_size must be positive or None")
    lots = math.floor(abs(quantity) / lot_size + 1e-9)
    return math.copysign(lots * lot_size, quantity) if lots else 0.0


def fixed_fractional_quantity(
    equity: float,
    price: float,
    fraction: float,
    *,
    lot_size: float | None = 1.0,
) -> float:
    """Quantity worth ``fraction`` of ``equity`` at ``price`` (signed like ``fraction``)."""
    if price <= 0 or not math.isfinite(price):
        raise ConfigError("price must be positive and finite")
    return round_to_lot(equity * fraction / price, lot_size)


def fixed_risk_quantity(
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_fraction: float,
    *,
    lot_size: float | None = 1.0,
) -> float:
    """Quantity such that hitting ``stop_price`` loses ``risk_fraction`` of equity.

    The sign follows the trade direction implied by the stop: a stop below the entry is
    a long, a stop above the entry is a short.
    """
    risk_per_unit = entry_price - stop_price
    if risk_per_unit == 0:
        raise ConfigError("stop_price must differ from entry_price")
    return round_to_lot(equity * risk_fraction / risk_per_unit, lot_size)


def realized_volatility(
    returns: npt.ArrayLike,
    periods_per_year: float = 252.0,
) -> npt.NDArray[np.float64]:
    """Annualised sample volatility of each column of ``returns`` (NaNs ignored)."""
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    count = np.sum(~np.isnan(arr), axis=0)
    out = np.full(arr.shape[1], np.nan)
    ok = count >= 2
    if ok.any():
        out[ok] = np.nanstd(arr[:, ok], axis=0, ddof=1) * math.sqrt(periods_per_year)
    return out


def volatility_target_weights(
    returns: npt.ArrayLike,
    target_volatility: float,
    *,
    periods_per_year: float = 252.0,
    max_weight: float | None = None,
    min_volatility: float = 1e-4,
) -> npt.NDArray[np.float64]:
    """Per-asset weights that scale each asset to ``target_volatility`` (annualised).

    ``weight_i = target_volatility / sigma_i`` with ``sigma_i`` the realized volatility
    of column ``i``. Assets with too little history get a weight of 0.
    """
    if target_volatility <= 0:
        raise ConfigError("target_volatility must be positive")
    vol = realized_volatility(returns, periods_per_year)
    weights = np.where(np.isnan(vol), 0.0, target_volatility / np.maximum(vol, min_volatility))
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
    return weights


def scale_to_gross(weights: Mapping[str, float], max_gross: float) -> dict[str, float]:
    """Scale weights down proportionally so that ``sum(|w|) <= max_gross``."""
    gross = sum(abs(w) for w in weights.values())
    if gross <= max_gross or gross == 0:
        return dict(weights)
    k = max_gross / gross
    return {s: w * k for s, w in weights.items()}


def rebalance_quantities(
    target_weights: Mapping[str, float],
    current_quantities: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    *,
    lot_size: float | None = 1.0,
    threshold: float = 0.0,
    min_notional: float = 0.0,
) -> dict[str, float]:
    """Signed order quantities that move the portfolio to ``target_weights``.

    Args:
        target_weights: Desired weight per symbol (fraction of ``equity``).
        current_quantities: Currently held signed quantities.
        prices: Prices used for sizing (typically the latest close).
        equity: Portfolio equity used for sizing.
        lot_size: Round order quantities to multiples of this (``None`` = fractional).
        threshold: Rebalancing band - skip symbols whose weight deviates from target by
            no more than this (absolute weight). Moves to or from exactly zero are
            always executed so positions are opened and closed on schedule.
        min_notional: Skip trades smaller than this value.

    Returns:
        Mapping ``symbol -> signed quantity`` (only non-zero trades are included).
    """
    orders: dict[str, float] = {}
    symbols = set(target_weights) | {s for s, q in current_quantities.items() if q != 0}
    for symbol in sorted(symbols):
        price = prices.get(symbol, math.nan)
        if not math.isfinite(price) or price <= 0:
            continue
        target_w = target_weights.get(symbol, 0.0)
        current_q = current_quantities.get(symbol, 0.0)
        current_w = current_q * price / equity if equity > 0 else 0.0
        closing = target_w == 0 and current_q != 0
        opening = current_q == 0 and target_w != 0
        if not (closing or opening) and abs(target_w - current_w) <= threshold:
            continue
        if closing:
            delta = -current_q
        else:
            target_q = round_to_lot(target_w * equity / price, lot_size)
            delta = target_q - current_q
        if delta == 0 or abs(delta) * price < min_notional:
            continue
        orders[symbol] = delta
    return orders
