"""A reusable, immutable description of how to run a backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from backtester.data.feed import DataFeed
from backtester.engine import BacktestResult, Engine
from backtester.execution.broker import IntrabarPath, SimulatedBroker
from backtester.execution.commission import CommissionModel, NoCommission
from backtester.execution.slippage import NoSlippage, SlippageModel
from backtester.orders import OrderType
from backtester.risk import RiskLimits, RiskManager
from backtester.strategy.base import Strategy

Bound = int | pd.Timestamp | str | None


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Execution, risk and accounting settings shared by many runs.

    Brokers and risk managers are stateful, so research tools (grid search,
    walk-forward) use this config to build fresh ones for every run. See
    :class:`~backtester.execution.SimulatedBroker` for ``intrabar`` and
    :class:`~backtester.risk.RiskManager` for ``target_order_type``.
    """

    initial_cash: float = 1_000_000.0
    slippage: SlippageModel = field(default_factory=NoSlippage)
    commission: CommissionModel = field(default_factory=NoCommission)
    max_participation: float | None = None
    strict_limits: bool = False
    intrabar: IntrabarPath = "worst"
    limits: RiskLimits = field(default_factory=RiskLimits)
    lot_size: float | None = 1.0
    rebalance_threshold: float = 0.0
    min_trade_notional: float = 0.0
    target_order_type: OrderType = OrderType.MARKET
    borrow_rate: float = 0.0
    periods_per_year: float = 252.0

    def build(
        self,
        feed: DataFeed,
        strategy: Strategy,
        *,
        start: Bound = None,
        end: Bound = None,
        check_invariants: bool = False,
    ) -> Engine:
        """Create an :class:`~backtester.Engine` with fresh broker and risk manager."""
        broker = SimulatedBroker(
            self.slippage,
            self.commission,
            max_participation=self.max_participation,
            strict_limits=self.strict_limits,
            intrabar=self.intrabar,
        )
        risk = RiskManager(
            self.limits,
            lot_size=self.lot_size,
            rebalance_threshold=self.rebalance_threshold,
            min_trade_notional=self.min_trade_notional,
            target_order_type=self.target_order_type,
        )
        return Engine(
            feed,
            strategy,
            initial_cash=self.initial_cash,
            broker=broker,
            risk=risk,
            periods_per_year=self.periods_per_year,
            borrow_rate=self.borrow_rate,
            start=start,
            end=end,
            check_invariants=check_invariants,
        )

    def run(
        self,
        feed: DataFeed,
        strategy: Strategy,
        *,
        start: Bound = None,
        end: Bound = None,
        check_invariants: bool = False,
    ) -> BacktestResult:
        """Build an engine and run it."""
        return self.build(
            feed, strategy, start=start, end=end, check_invariants=check_invariants
        ).run()
