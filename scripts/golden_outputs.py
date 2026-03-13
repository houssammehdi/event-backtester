"""Hash every output of 25 diverse backtests, to check that a change alters no result.

Run it before and after a change and compare the two files::

    python scripts/golden_outputs.py before.json
    ... change the code ...
    python scripts/golden_outputs.py after.json && diff before.json after.json

Each run's digest covers the equity, cash, exposure, positions, weights and price
series (their exact bytes), the fill table, every order record and the risk log. The
runs cover every built-in strategy, the four intrabar policies, participation caps,
strict limits, kill switches, warm-up windows, short borrow costs, MOC rebalancing,
missing bars, and :class:`RandomOrders`, which submits random orders of every type
(brackets, OCO pairs, trailing stops, auctions), cancels and targets. Digests depend
on the NumPy build, so compare files from the same environment only.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from backtester import (
    BpsCommission,
    DataFeed,
    FixedBpsSlippage,
    MarketView,
    OrderType,
    RiskLimits,
    SquareRootImpactSlippage,
    Strategy,
    StrategyContext,
    TimeInForce,
    generate_multi_asset,
    generate_ohlcv,
)
from backtester.config import BacktestConfig
from backtester.engine import BacktestResult
from backtester.strategies import (
    AllocationStrategy,
    BollingerMeanReversion,
    CrossSectionalMomentum,
    SmaCrossover,
    TimeSeriesMomentum,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from intrabar_sensitivity import BracketBreakout


class RandomOrders(Strategy):
    """Random orders of every type, brackets, OCO pairs, cancels and targets."""

    name = "random_orders"

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        """Draw one action per traded symbol, and now and then a partial target."""
        rng = self.rng
        for symbol in view.symbols:
            if view.has_bar(symbol):
                self._act(symbol, view.price(symbol), ctx)
        if rng.random() < 0.02:
            weights = {s: float(rng.uniform(-0.2, 0.3)) for s in view.symbols if rng.random() < 0.6}
            ctx.target_weights(weights, partial=bool(rng.random() < 0.5))
        _ = ctx.equity, ctx.cash, ctx.weights(), ctx.positions()

    def _act(self, symbol: str, price: float, ctx: StrategyContext) -> None:
        rng = self.rng
        u = rng.random()
        qty = float(rng.integers(1, 300))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        off = float(rng.uniform(0.002, 0.04))
        tif = TimeInForce.GTC if rng.random() < 0.5 else TimeInForce.DAY
        buy = sign > 0
        near = price * (1 - off) if buy else price * (1 + off)  # limit side
        far = price * (1 + off) if buy else price * (1 - off)  # stop side
        if u < 0.06:
            ctx.order(symbol, sign * qty, tif=tif)
        elif u < 0.12:
            ctx.order(symbol, sign * qty, OrderType.LIMIT, limit_price=near, tif=tif)
        elif u < 0.18:
            ctx.order(symbol, sign * qty, OrderType.STOP, stop_price=far, tif=tif)
        elif u < 0.22:
            limit = far * (1 + (rng.random() - 0.5) * 0.02)
            ctx.order(
                symbol, sign * qty, OrderType.STOP_LIMIT, stop_price=far, limit_price=limit, tif=tif
            )
        elif u < 0.26:
            trail = {"trail_percent": off} if rng.random() < 0.5 else {"trail_amount": price * off}
            ctx.order(symbol, sign * qty, OrderType.TRAILING_STOP, tif=tif, **trail)
        elif u < 0.28:
            ctx.order(symbol, sign * qty, OrderType.MARKET_ON_OPEN)
        elif u < 0.30:
            ctx.order(symbol, sign * qty, OrderType.MARKET_ON_CLOSE)
        elif u < 0.36:
            self._bracket(symbol, sign * qty, price, off, ctx)
        elif u < 0.40:
            group = ctx.oco_group()
            ctx.order(
                symbol, sign * qty, OrderType.LIMIT, limit_price=near, oco_group=group, tif=tif
            )
            ctx.order(symbol, sign * qty, OrderType.STOP, stop_price=far, oco_group=group, tif=tif)
        elif u < 0.43:
            orders = ctx.open_orders(symbol)
            if orders:
                ctx.cancel(orders[int(rng.integers(0, len(orders)))].id)
        elif u < 0.44:
            ctx.cancel(symbol=symbol)
        elif u < 0.47 and ctx.position(symbol) != 0:
            ctx.order(symbol, -ctx.position(symbol))

    def _bracket(
        self, symbol: str, qty: float, price: float, off: float, ctx: StrategyContext
    ) -> None:
        rng = self.rng
        buy = qty > 0
        stop_loss = price * (1 - 2 * off) if buy else price * (1 + 2 * off)
        take_profit = price * (1 + 3 * off) if buy else price * (1 - 3 * off)
        entry: dict[str, Any] = {}
        kind = int(rng.integers(0, 3))
        if kind == 1:
            entry = {"entry_type": OrderType.LIMIT}
            entry["limit_price"] = price * (1 - off / 2) if buy else price * (1 + off / 2)
        elif kind == 2:
            entry = {"entry_type": OrderType.STOP}
            entry["stop_price"] = price * (1 + off / 2) if buy else price * (1 - off / 2)
        exits = int(rng.integers(0, 4))
        if exits == 0:
            ctx.bracket(symbol, qty, stop_loss=stop_loss, take_profit=take_profit, **entry)
        elif exits == 1:
            ctx.bracket(symbol, qty, trail_percent=off, take_profit=take_profit, **entry)
        elif exits == 2:
            ctx.bracket(symbol, qty, take_profit=take_profit, **entry)
        else:
            ctx.bracket(symbol, qty, trail_amount=price * off, **entry)


def digest(result: BacktestResult) -> dict[str, Any]:
    """SHA-256 over every output of a run, plus a few readable numbers."""
    h = hashlib.sha256()
    for series in (result.equity, result.cash):
        h.update(series.to_numpy().tobytes())
    h.update(result.equity.index.asi8.tobytes())
    for frame in (result.exposure, result.positions, result.weights, result.prices):
        h.update(frame.to_numpy().tobytes())
        h.update(repr(list(frame.columns)).encode())
    h.update(result.fills.to_csv(float_format="%r").encode())
    for record in (*result.orders, *result.risk_log):
        h.update(repr(record).encode())
    totals = (result.halted_at, result.total_commission, result.total_slippage)
    h.update(repr((*totals, result.total_borrow_cost)).encode())
    return {
        "sha256": h.hexdigest(),
        "final_equity": repr(result.final_equity),
        "fills": len(result.fills),
        "orders": len(result.orders),
    }


Run = tuple[str, BacktestConfig, DataFeed, Callable[[], Strategy], dict[str, Any]]


def runs() -> list[Run]:
    """The 25 golden backtests."""
    feed5 = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    feed20 = DataFeed(generate_ohlcv(n_symbols=20, years=6, seed=7))
    multi = DataFeed(generate_multi_asset(years=8, seed=3, missing_prob=0.02))
    costs = {"slippage": FixedBpsSlippage(2.0), "commission": BpsCommission(1.0)}
    base = BacktestConfig(**costs, max_participation=0.1)  # type: ignore[arg-type]
    out: list[Run] = [
        ("tsmom", base, feed5, TimeSeriesMomentum, {}),
        ("tsmom_checked", base, feed5, TimeSeriesMomentum, {"check_invariants": True}),
        ("sma_daily", base, feed20, lambda: SmaCrossover(20, 100, rebalance="daily"), {}),
        (
            "sma_short_borrow",
            BacktestConfig(borrow_rate=0.03),
            feed20,
            lambda: SmaCrossover(10, 50, allow_short=True),
            {"start": 300},
        ),
        ("xsmom_window", base, feed20, CrossSectionalMomentum, {"start": 600, "end": -50}),
        (
            "bollinger_impact",
            BacktestConfig(slippage=SquareRootImpactSlippage(), max_participation=0.01),
            feed5,
            BollingerMeanReversion,
            {},
        ),
        ("hrp_missing_bars", base, multi, lambda: AllocationStrategy("hrp"), {}),
        (
            "erc_moc",
            BacktestConfig(target_order_type=OrderType.MARKET_ON_CLOSE),
            multi,
            lambda: AllocationStrategy("erc"),
            {},
        ),
        (
            "kill_switch",
            BacktestConfig(limits=RiskLimits(max_drawdown=0.08)),
            feed5,
            lambda: TimeSeriesMomentum(target_vol=0.6),
            {},
        ),
    ]
    tight = functools.partial(BracketBreakout, stop_atr=0.5, target_atr=0.75)
    for policy in ("worst", "best", "high_first", "low_first"):
        cfg = BacktestConfig(**costs, max_participation=0.1, intrabar=policy)  # type: ignore[arg-type]
        out.append((f"bracket_{policy}", cfg, feed5, tight, {}))
    random_configs = {
        "random_worst": BacktestConfig(
            slippage=FixedBpsSlippage(3.0), commission=BpsCommission(1.0)
        ),
        "random_capacity": BacktestConfig(
            slippage=FixedBpsSlippage(1.0),
            max_participation=0.00005,
            limits=RiskLimits(max_position_weight=0.3, max_gross_leverage=1.5),
        ),
        "random_strict_best": BacktestConfig(
            strict_limits=True, intrabar="best", max_participation=0.0001
        ),
        "random_low_long_only": BacktestConfig(
            intrabar="low_first", limits=RiskLimits(allow_short=False)
        ),
        "random_high_kill": BacktestConfig(
            intrabar="high_first", limits=RiskLimits(max_drawdown=0.03)
        ),
    }
    for seed, (name, cfg) in enumerate(random_configs.items()):
        out.append((name, cfg, feed5, lambda s=seed: RandomOrders(s), {"check_invariants": True}))
        out.append((f"{name}_multi", cfg, multi, lambda s=seed: RandomOrders(s + 100), {}))
    return out


def main() -> None:
    """Run every golden backtest and write the digests as JSON."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/golden_outputs.py OUT.json")
    digests = {}
    for name, cfg, feed, factory, kwargs in runs():
        digests[name] = digest(cfg.run(feed, factory(), **kwargs))
        print(name, digests[name]["sha256"][:16], digests[name]["final_equity"], flush=True)
    Path(sys.argv[1]).write_text(json.dumps(digests, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
