"""How much does a bracket strategy depend on what happened inside the bar?

A daily bar does not say whether its high came before its low. When a position's stop
and its target both lie inside one bar, the backtest has to assume an order. This
script runs a breakout strategy with bracket orders (a buy-stop entry at the 20-day
high, a stop-loss and a target at multiples of the average range) under the four
intrabar policies of the broker, for a wide and a tight bracket, and counts the exits
that fell on ambiguous bars.

``worst`` (the default) and ``best`` bound what any path assumption could give;
``high_first`` / ``low_first`` are fixed conventions. The spread between the bounds is
the part of the result that daily data cannot resolve.

Run:  python examples/intrabar_sensitivity.py
"""

from __future__ import annotations

import numpy as np

from backtester import (
    BpsCommission,
    DataFeed,
    FixedBpsSlippage,
    MarketView,
    OrderType,
    Strategy,
    StrategyContext,
    generate_ohlcv,
)
from backtester.analytics.report import num, pct, table
from backtester.config import BacktestConfig
from backtester.engine import BacktestResult
from backtester.portfolio import fixed_fractional_quantity, fixed_risk_quantity

POLICIES = ("worst", "low_first", "high_first", "best")


class BracketBreakout(Strategy):
    """Buy-stop breakouts protected by a bracket (stop-loss and target at ATR multiples)."""

    name = "bracket_breakout"

    def __init__(
        self,
        entry: int = 20,
        stop_atr: float = 2.0,
        target_atr: float = 3.0,
        risk_fraction: float = 0.01,
        max_weight: float = 0.25,
    ) -> None:
        self.entry = entry
        self.stop_atr = stop_atr
        self.target_atr = target_atr
        self.risk_fraction = risk_fraction
        self.max_weight = max_weight

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        """Place a bracketed breakout entry for every flat symbol without orders."""
        for symbol in view.symbols:
            if ctx.position(symbol) != 0 or ctx.open_orders(symbol) or not view.has_bar(symbol):
                continue
            highs = view.series(symbol, "high", self.entry)
            lows = view.series(symbol, "low", self.entry)
            if len(highs) < self.entry:
                continue
            breakout = float(highs.max())
            atr = float(np.mean(highs - lows))
            stop = breakout - self.stop_atr * atr
            qty = min(
                fixed_risk_quantity(ctx.equity, breakout, stop, self.risk_fraction),
                fixed_fractional_quantity(ctx.equity, breakout, self.max_weight),
            )
            if qty > 0:
                ctx.bracket(
                    symbol,
                    qty,
                    OrderType.STOP,
                    stop_price=breakout,
                    stop_loss=stop,
                    take_profit=breakout + self.target_atr * atr,
                )


def ambiguous_exits(result: BacktestResult, feed: DataFeed) -> int:
    """Exits on bars whose range contained both the stop-loss and the target."""
    by_id = {o.id: o for o in result.orders}
    count = 0
    for order in result.orders:
        if order.tag != "stop_loss" or order.parent_id is None:
            continue
        target = next(
            (by_id[i] for i in by_id if by_id[i].parent_id == order.parent_id and i != order.id),
            None,
        )
        exit_order = order if order.filled_quantity > 0 else target
        if target is None or exit_order is None or exit_order.filled_quantity == 0:
            continue
        assert exit_order.closed_at is not None
        bar = feed.view(int(feed.index.get_loc(exit_order.closed_at))).bar(order.symbol)
        assert order.stop_price is not None
        assert target.limit_price is not None
        if bar is not None and bar.low <= order.stop_price and bar.high >= target.limit_price:
            count += 1
    return count


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    header = ("intrabar", "trades", "hit rate", "CAGR", "Sharpe", "max DD", "ambiguous exits")
    for stop_atr, target_atr in ((2.0, 3.0), (0.5, 0.75)):
        rows = []
        for policy in POLICIES:
            config = BacktestConfig(
                slippage=FixedBpsSlippage(2.0),
                commission=BpsCommission(1.0),
                max_participation=0.1,
                intrabar=policy,  # type: ignore[arg-type]
            )
            strategy = BracketBreakout(stop_atr=stop_atr, target_atr=target_atr)
            result = config.run(feed, strategy, check_invariants=True)
            m = result.metrics()
            rows.append(
                (
                    policy,
                    str(m.trades.n_trades),
                    pct(m.trades.hit_rate, 1),
                    pct(m.cagr),
                    num(m.sharpe),
                    pct(m.max_drawdown),
                    str(ambiguous_exits(result, feed)),
                )
            )
        print(
            f"Bracket breakout: 20-day high entry, stop {stop_atr} ATR, target {target_atr} "
            "ATR (5 symbols, 10 years, 2 + 1 bps costs)"
        )
        print(table(rows, header=header))
        print()


if __name__ == "__main__":
    main()
