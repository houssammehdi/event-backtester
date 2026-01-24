"""Write your own strategy: a Donchian breakout that trades with stop orders.

Unlike the built-in target-weight strategies, this one uses the explicit order API:

* entries are *buy-stop* orders at the highest high of the last ``entry`` bars, so the
  position is only opened if the next bar actually breaks out (gap-ups fill at the
  open, a worse price - the broker models that);
* each position is sized so that a move of ``stop_atr`` average ranges costs
  ``risk_fraction`` of equity (fixed-risk sizing), capped at ``max_weight`` of equity;
* exits are *sell-stop* orders at the lowest low of the last ``exit`` bars, re-placed
  every bar as a trailing stop.

Run:  python examples/custom_strategy.py
"""

from backtester import (
    DataFeed,
    Engine,
    MarketView,
    OrderType,
    PerShareCommission,
    RiskLimits,
    SimulatedBroker,
    SquareRootImpactSlippage,
    Strategy,
    StrategyContext,
    TimeInForce,
    generate_ohlcv,
)
from backtester.portfolio import fixed_fractional_quantity, fixed_risk_quantity


class DonchianBreakout(Strategy):
    """Long-only channel breakout with ATR-based position sizing and trailing stops."""

    name = "donchian"

    def __init__(
        self,
        entry: int = 55,
        exit: int = 20,
        risk_fraction: float = 0.01,
        stop_atr: float = 2.0,
        max_weight: float = 0.2,
    ) -> None:
        self.entry = entry
        self.exit = exit
        self.risk_fraction = risk_fraction
        self.stop_atr = stop_atr
        self.max_weight = max_weight

    def on_bar(self, view: MarketView, ctx: StrategyContext) -> None:
        """Place breakout entries for flat symbols and trail stops for open positions."""
        for symbol in view.symbols:
            highs = view.series(symbol, "high", self.entry)
            lows = view.series(symbol, "low", self.entry)
            if len(highs) < self.entry or not view.has_bar(symbol):
                continue
            position = ctx.position(symbol)
            if position == 0:
                breakout = float(highs.max())
                atr = float((highs[-20:] - lows[-20:]).mean())
                qty = min(
                    fixed_risk_quantity(
                        ctx.equity, breakout, breakout - self.stop_atr * atr, self.risk_fraction
                    ),
                    fixed_fractional_quantity(ctx.equity, breakout, self.max_weight),
                )
                if qty > 0:
                    ctx.order(symbol, qty, OrderType.STOP, stop_price=breakout, tag="entry")
            elif position > 0:
                ctx.cancel(symbol=symbol)
                trailing = float(lows[-self.exit :].min())
                ctx.order(
                    symbol,
                    -position,
                    OrderType.STOP,
                    stop_price=trailing,
                    tif=TimeInForce.GTC,
                    tag="exit",
                )


def main() -> None:
    feed = DataFeed(generate_ohlcv(n_symbols=5, years=10, seed=42))
    broker = SimulatedBroker(
        slippage=SquareRootImpactSlippage(eta=0.5, spread_bps=2.0),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    result = Engine(
        feed,
        DonchianBreakout(),
        broker=broker,
        risk=RiskLimits(max_position_weight=0.25, max_gross_leverage=1.0),
        check_invariants=True,
    ).run()
    print(result.report())
    trades = result.trades()
    print(f"\nLast five round trips ({len(trades)} total):")
    print(trades.tail(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
