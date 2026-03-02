"""Trailing stops, brackets, OCO groups, MOO/MOC and the intrabar path policy.

Bars are written ``(open, high, low, close, volume)``. Path A is open-high-low-close
("high_first"), path B open-low-high-close ("low_first"); the default policy keeps the
path whose fills leave the lower equity at the bar's close.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtester import (
    ConfigError,
    DataFeed,
    FixedBpsSlippage,
    OrderError,
    OrderEvent,
    OrderStatus,
    OrderType,
    Side,
    SimulatedBroker,
    TimeInForce,
)
from backtester.execution import IntrabarPath
from tests.conftest import Row, bars

BUY, SELL = Side.BUY, Side.SELL
GTC = TimeInForce.GTC


def feed_of(*rows: Row) -> DataFeed:
    """Bar 0 (the decision bar, everything at 100) followed by the tested bars."""
    return DataFeed({"A": bars([(100, 100, 100, 100, 1e6), *rows])})


def submit(
    broker: SimulatedBroker,
    feed: DataFeed,
    order_id: int,
    side: Side,
    qty: float,
    kind: OrderType = OrderType.MARKET,
    **kw: Any,
) -> None:
    broker.submit(
        OrderEvent(
            feed.index[0],
            order_id=order_id,
            symbol="A",
            side=side,
            quantity=qty,
            order_type=kind,
            **kw,
        )
    )


def run_bar(
    broker: SimulatedBroker, feed: DataFeed, pos: int = 1
) -> list[tuple[int, float, float]]:
    return [
        (f.order_id, f.quantity, round(f.price, 10)) for f in broker.process_bar(feed.view(pos))
    ]


def long_bracket(
    broker: SimulatedBroker,
    feed: DataFeed,
    *,
    entry: OrderType = OrderType.MARKET,
    entry_tif: TimeInForce = TimeInForce.DAY,
    qty: float = 100,
    stop: float = 95.0,
    target: float = 110.0,
    **entry_kw: Any,
) -> None:
    """Entry 1 (buy), stop-loss 2 (sell stop), take-profit 3 (sell limit)."""
    submit(broker, feed, 1, BUY, qty, entry, tif=entry_tif, **entry_kw)
    submit(broker, feed, 2, SELL, qty, OrderType.STOP, stop_price=stop, tif=GTC, parent_id=1)
    submit(broker, feed, 3, SELL, qty, OrderType.LIMIT, limit_price=target, tif=GTC, parent_id=1)


def status(broker: SimulatedBroker, *ids: int) -> list[OrderStatus]:
    return [broker.get(i).status for i in ids]


# ---------------------------------------------------------------------- trailing stops
class TestTrailingStop:
    def test_trails_the_high_and_fills_at_the_level(self) -> None:
        # ref 100 -> 106 on bar 1 (level 100.7); bar 2 falls through 100.7 intrabar
        feed = feed_of((102, 106, 101, 105, 1e6), (104, 105, 100, 101, 1e6))
        broker = SimulatedBroker()
        submit(
            broker,
            feed,
            1,
            SELL,
            10,
            OrderType.TRAILING_STOP,
            trail_percent=0.05,
            trail_reference=100.0,
            tif=GTC,
        )
        assert run_bar(broker, feed, 1) == []
        assert broker.get(1).trail_reference == 106
        assert run_bar(broker, feed, 2) == [(1, 10, pytest.approx(100.7))]

    def test_gap_below_the_level_fills_at_the_open(self) -> None:
        """3.00 trail from 100. Bar 1 (100, 101, 99, 100.5) ratchets the level to 98
        without reaching it on either path; bar 2 opens at 96, below it."""
        feed = feed_of((100, 101, 99, 100.5, 1e6), (96, 97, 95, 96.5, 1e6))
        broker = SimulatedBroker()
        submit(
            broker,
            feed,
            1,
            SELL,
            10,
            OrderType.TRAILING_STOP,
            trail_amount=3.0,
            trail_reference=100.0,
            tif=GTC,
        )
        assert run_bar(broker, feed, 1) == []
        assert broker.get(1).trail_level() == 98
        assert run_bar(broker, feed, 2) == [(1, 10, 96)]  # the (worse) open

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            ("worst", [(1, 10, 104.5)]),
            ("high_first", [(1, 10, 104.5)]),
            ("best", []),
            ("low_first", []),
        ],
    )
    def test_intrabar_ratchet_depends_on_the_path(
        self, policy: IntrabarPath, expected: list[tuple[int, float, float]]
    ) -> None:
        """Bar (105, 110, 103, 108), 5% trail from 100.

        Path A: the high (110) comes first, the level ratchets to 104.5 and the fall to
        103 crosses it: sell at 104.5, worth 10 * (104.5 - 108) = -35 at the close.
        Path B: the low (103) comes first while the level is still 99.75; then the
        high, and the close (108) stays above 104.5: no fill, worth 0. The worst case
        is path A.
        """
        feed = feed_of((105, 110, 103, 108, 1e6))
        broker = SimulatedBroker(intrabar=policy)
        submit(
            broker,
            feed,
            1,
            SELL,
            10,
            OrderType.TRAILING_STOP,
            trail_percent=0.05,
            trail_reference=100.0,
            tif=GTC,
        )
        assert run_bar(broker, feed) == expected
        assert broker.get(1).trail_reference == 110  # both paths end having seen the high

    def test_buy_trailing_stop_trails_the_low(self) -> None:
        """Short protection, 3% from 100. Bar 1 (97, 97.5, 95, 96) takes the low to 95
        (level 97.85) and never trades back above it; bar 2 rallies through 97.85."""
        feed = feed_of((97, 97.5, 95, 96, 1e6), (96, 99, 95.5, 98.5, 1e6))
        broker = SimulatedBroker()
        submit(
            broker,
            feed,
            1,
            BUY,
            10,
            OrderType.TRAILING_STOP,
            trail_percent=0.03,
            trail_reference=100.0,
            tif=GTC,
        )
        assert run_bar(broker, feed, 1) == []
        assert broker.get(1).trail_reference == 95
        assert run_bar(broker, feed, 2) == [(1, 10, pytest.approx(97.85))]

    @pytest.mark.parametrize(("policy", "price"), [("worst", 99.5), ("high_first", 100.5)])
    def test_without_a_reference_it_starts_at_the_open(
        self, policy: IntrabarPath, price: float
    ) -> None:
        """0.50 trail, no reference: it starts at the open, 100 (level 99.5). Path B
        falls to 99 first and sells at 99.5 (worth -5 at the 100 close); path A first
        ratchets to 101 and sells at 100.5 on the way down (worth +5)."""
        feed = feed_of((100, 101, 99, 100, 1e6))
        broker = SimulatedBroker(intrabar=policy)
        submit(broker, feed, 1, SELL, 10, OrderType.TRAILING_STOP, trail_amount=0.5, tif=GTC)
        assert run_bar(broker, feed) == [(1, 10, price)]

    def test_validation(self) -> None:
        with pytest.raises(OrderError, match="exactly one"):
            OrderEvent(
                pd.Timestamp("2024-01-02"),
                order_id=1,
                symbol="A",
                side=SELL,
                quantity=1,
                order_type=OrderType.TRAILING_STOP,
            )
        with pytest.raises(OrderError, match=r"\(0, 1\)"):
            OrderEvent(
                pd.Timestamp("2024-01-02"),
                order_id=1,
                symbol="A",
                side=SELL,
                quantity=1,
                order_type=OrderType.TRAILING_STOP,
                trail_percent=1.5,
            )
        with pytest.raises(OrderError, match="must not set a trail"):
            OrderEvent(
                pd.Timestamp("2024-01-02"),
                order_id=1,
                symbol="A",
                side=SELL,
                quantity=1,
                trail_amount=1.0,
            )


# ---------------------------------------------------------------------- brackets
class TestBracket:
    def test_exits_arm_on_entry_and_the_target_cancels_the_stop(self) -> None:
        feed = feed_of((100, 104, 97, 103, 1e6), (103, 111, 101, 108, 1e6))
        broker = SimulatedBroker()
        long_bracket(broker, feed)
        assert not broker.get(2).armed
        assert run_bar(broker, feed, 1) == [(1, 100, 100)]  # entry at the open
        assert broker.get(2).armed and broker.get(3).armed  # noqa: PT018
        assert status(broker, 2, 3) == [OrderStatus.NEW, OrderStatus.NEW]
        assert run_bar(broker, feed, 2) == [(3, 100, 110)]
        assert status(broker, 1, 2, 3) == [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.FILLED,
        ]
        assert broker.open_orders() == []

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [("worst", (2, 95)), ("low_first", (2, 95)), ("best", (3, 110)), ("high_first", (3, 110))],
    )
    def test_stop_and_target_in_one_bar(
        self, policy: IntrabarPath, expected: tuple[int, float]
    ) -> None:
        """Bar 2 (100, 112, 94, 105) trades through both exits.

        Path A reaches the target first (sell at 110, worth +500 at the 105 close); path
        B the stop (sell at 95, worth -1000). The worst case takes the stop.
        """
        feed = feed_of((100, 101, 99, 100, 1e6), (100, 112, 94, 105, 1e6))
        broker = SimulatedBroker(intrabar=policy)
        long_bracket(broker, feed)
        run_bar(broker, feed, 1)
        oid, price = expected
        assert run_bar(broker, feed, 2) == [(oid, 100, price)]
        other = 3 if oid == 2 else 2
        assert broker.get(other).status is OrderStatus.CANCELLED

    def test_submitted_orders_stay_the_live_records(self) -> None:
        """The worst-case policy simulates both paths on copies; the chosen outcome is
        written back, so the objects submit() returned (and the book) stay current."""
        feed = feed_of((100, 101, 99, 100, 1e6), (100, 112, 94, 105, 1e6))
        broker = SimulatedBroker()
        records = [
            broker.submit(
                OrderEvent(feed.index[0], order_id=1, symbol="A", side=BUY, quantity=100)
            ),
            broker.submit(
                OrderEvent(
                    feed.index[0],
                    order_id=2,
                    symbol="A",
                    side=SELL,
                    quantity=100,
                    order_type=OrderType.STOP,
                    stop_price=95.0,
                    tif=GTC,
                    parent_id=1,
                )
            ),
        ]
        run_bar(broker, feed, 1)
        run_bar(broker, feed, 2)
        assert [broker.get(i) for i in (1, 2)] == records
        assert all(broker.get(o.id) is o for o in records)
        assert records[1].status is OrderStatus.FILLED
        assert records[1].avg_fill_price == 95.0

    def test_gap_through_the_stop_fills_at_the_open(self) -> None:
        feed = feed_of((100, 101, 99, 100, 1e6), (90, 92, 88, 91, 1e6))
        broker = SimulatedBroker()
        long_bracket(broker, feed)
        run_bar(broker, feed, 1)
        assert run_bar(broker, feed, 2) == [(2, 100, 90)]  # the worse price: the open
        assert broker.get(3).status is OrderStatus.CANCELLED

    def test_gap_through_the_target_fills_at_the_better_open(self) -> None:
        feed = feed_of((100, 101, 99, 100, 1e6), (115, 116, 113, 114, 1e6))
        broker = SimulatedBroker()
        long_bracket(broker, feed)
        run_bar(broker, feed, 1)
        assert run_bar(broker, feed, 2) == [(3, 100, 115)]

    def test_round_trip_inside_the_entry_bar(self) -> None:
        # entry at the open (100), then the low (94) takes out the stop at 95
        feed = feed_of((100, 102, 94, 97, 1e6))
        broker = SimulatedBroker()
        long_bracket(broker, feed)
        assert run_bar(broker, feed) == [(1, 100, 100), (2, 100, 95)]

    @pytest.mark.parametrize(
        ("policy", "fills"),
        [
            ("worst", [(1, 100, 98)]),
            ("best", [(1, 100, 98), (3, 100, 103)]),
        ],
    )
    def test_limit_entry_and_target_in_one_bar(
        self, policy: IntrabarPath, fills: list[tuple[int, float, float]]
    ) -> None:
        """Buy limit 98, stop 93, target 103, bar (100, 104, 94, 101).

        Path A: the high (104) comes before the entry, which fills at 98 on the way down;
        the target is never revisited: long from 98, worth +300 at the 101 close. Path B:
        the entry fills at 98 on the way down to 94, then the rally reaches 103: a round
        trip worth +500. The worst case keeps the position open.
        """
        feed = feed_of((100, 104, 94, 101, 1e6))
        broker = SimulatedBroker(intrabar=policy)
        long_bracket(broker, feed, entry=OrderType.LIMIT, limit_price=98.0, stop=93.0, target=103.0)
        assert run_bar(broker, feed) == fills

    def test_partial_entry_sizes_the_exits(self) -> None:
        # 10% of 1,000 shares a bar: the DAY entry for 300 fills 100 and expires
        feed = feed_of((100, 101, 99, 100, 1_000), (100, 101, 94, 96, 1_000))
        broker = SimulatedBroker(max_participation=0.1)
        long_bracket(broker, feed, qty=300)
        assert run_bar(broker, feed, 1) == [(1, 100, 100)]
        assert broker.get(1).status is OrderStatus.EXPIRED
        assert run_bar(broker, feed, 2) == [(2, 100, 95)]
        stop = broker.get(2)
        assert (stop.status, stop.quantity, stop.filled_quantity) == (OrderStatus.FILLED, 100, 100)
        assert broker.get(3).status is OrderStatus.CANCELLED
        assert broker.open_orders() == []

    def test_first_exit_fill_ends_the_rest_of_the_entry(self) -> None:
        # GTC entry for 300, 100 per bar; bar 2's low hits the stop after the open fill
        feed = feed_of((100, 101, 99, 100, 1_000), (99, 99.5, 94, 96, 10_000))
        broker = SimulatedBroker(max_participation=0.1)
        long_bracket(broker, feed, qty=300, entry_tif=GTC)
        run_bar(broker, feed, 1)
        # bar 2: 1,000 shares of capacity - the entry takes 200 at the open, the stop
        # then closes all 300
        assert run_bar(broker, feed, 2) == [(1, 200, 99), (2, 300, 95)]
        assert status(broker, 1, 2, 3) == [
            OrderStatus.FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        ]

    def test_unfilled_entry_cancels_its_exits(self) -> None:
        feed = feed_of((100, 101, 99, 100, 1e6))
        broker = SimulatedBroker()
        long_bracket(broker, feed, entry=OrderType.LIMIT, limit_price=90.0)
        assert run_bar(broker, feed) == []
        assert status(broker, 1, 2, 3) == [
            OrderStatus.EXPIRED,
            OrderStatus.CANCELLED,
            OrderStatus.CANCELLED,
        ]
        broker2 = SimulatedBroker()
        long_bracket(broker2, feed, entry=OrderType.LIMIT, limit_price=90.0, entry_tif=GTC)
        assert broker2.cancel(1, feed.index[0])
        assert status(broker2, 2, 3) == [OrderStatus.CANCELLED, OrderStatus.CANCELLED]

    def test_short_bracket_worst_case_is_the_stop_above(self) -> None:
        feed = feed_of((100, 101, 99, 100, 1e6), (100, 106, 89, 97, 1e6))
        broker = SimulatedBroker()
        submit(broker, feed, 1, SELL, 50, OrderType.MARKET)
        submit(broker, feed, 2, BUY, 50, OrderType.STOP, stop_price=105.0, tif=GTC, parent_id=1)
        submit(broker, feed, 3, BUY, 50, OrderType.LIMIT, limit_price=90.0, tif=GTC, parent_id=1)
        run_bar(broker, feed, 1)
        assert run_bar(broker, feed, 2) == [(2, 50, 105)]

    def test_trailing_exit_trails_from_the_entry_fill(self) -> None:
        """5% trailing exit armed at the entry's open fill (100). Bar 1 (100, 104, 99.5,
        103) lifts the level to 98.8 but stays above it on both paths; bar 2 falls
        through it."""
        feed = feed_of((100, 104, 99.5, 103, 1e6), (103, 103.5, 98, 99, 1e6))
        broker = SimulatedBroker()
        submit(broker, feed, 1, BUY, 10)
        submit(
            broker,
            feed,
            2,
            SELL,
            10,
            OrderType.TRAILING_STOP,
            trail_percent=0.05,
            tif=GTC,
            parent_id=1,
        )
        assert run_bar(broker, feed, 1) == [(1, 10, 100)]
        assert broker.get(2).trail_reference == 104
        assert run_bar(broker, feed, 2) == [(2, 10, pytest.approx(98.8))]

    def test_links_are_validated(self) -> None:
        feed = feed_of((100, 101, 99, 100, 1e6))
        broker = SimulatedBroker()
        with pytest.raises(OrderError, match="no entry"):
            submit(broker, feed, 2, SELL, 1, OrderType.STOP, stop_price=90.0, parent_id=1)


# ---------------------------------------------------------------------- OCO
class TestOCO:
    def test_first_fill_cancels_the_others(self) -> None:
        feed = feed_of((100, 106, 99, 104, 1e6))
        broker = SimulatedBroker()
        submit(broker, feed, 1, BUY, 10, OrderType.STOP, stop_price=105.0, oco_group=7)
        submit(broker, feed, 2, SELL, 10, OrderType.STOP, stop_price=95.0, oco_group=7)
        assert run_bar(broker, feed) == [(1, 10, 105)]
        assert broker.get(2).status is OrderStatus.CANCELLED

    def test_both_triggered_in_one_bar_resolves_to_the_worse(self) -> None:
        """Buy stop 105 / sell stop 95 on (100, 106, 94, 101): path A buys at 105 (worth
        -40 at the close), path B sells at 95 (worth -60): the worst case sells."""
        feed = feed_of((100, 106, 94, 101, 1e6))
        worst = SimulatedBroker()
        best = SimulatedBroker(intrabar="best")
        for broker in (worst, best):
            submit(broker, feed, 1, BUY, 10, OrderType.STOP, stop_price=105.0, oco_group=7)
            submit(broker, feed, 2, SELL, 10, OrderType.STOP, stop_price=95.0, oco_group=7)
        assert run_bar(worst, feed) == [(2, 10, 95)]
        assert run_bar(best, feed) == [(1, 10, 105)]

    def test_members_must_share_a_symbol(self) -> None:
        feed = DataFeed(
            {"A": bars([(100, 100, 100, 100, 1e6)] * 2), "B": bars([(50, 50, 50, 50, 1e6)] * 2)}
        )
        broker = SimulatedBroker()
        submit(broker, feed, 1, BUY, 1, OrderType.LIMIT, limit_price=99.0, oco_group=3)
        with pytest.raises(OrderError, match="same symbol"):
            broker.submit(
                OrderEvent(feed.index[0], order_id=2, symbol="B", side=BUY, quantity=1, oco_group=3)
            )


# ---------------------------------------------------------------------- auctions
class TestAuctions:
    def test_moc_decided_at_t_fills_at_the_close_of_t_plus_1(self) -> None:
        feed = feed_of((101, 103, 99, 102, 1e6))
        broker = SimulatedBroker(slippage=FixedBpsSlippage(10))
        submit(broker, feed, 1, BUY, 10, OrderType.MARKET_ON_CLOSE)
        assert run_bar(broker, feed, 0) == []  # never on the decision bar
        assert run_bar(broker, feed, 1) == [(1, 10, pytest.approx(102 * 1.001))]

    def test_fills_are_ordered_open_intrabar_close(self) -> None:
        feed = feed_of((101, 103, 98, 102, 1e6))
        broker = SimulatedBroker()
        submit(broker, feed, 1, BUY, 1, OrderType.MARKET_ON_CLOSE)
        submit(broker, feed, 2, BUY, 1, OrderType.LIMIT, limit_price=99.0)
        submit(broker, feed, 3, BUY, 1, OrderType.MARKET_ON_OPEN)
        assert [f[0] for f in run_bar(broker, feed)] == [3, 2, 1]

    def test_moc_expires_on_a_missing_bar_and_competes_for_capacity(self) -> None:
        a = bars([(100, 100, 100, 100, 1_000)] * 2)
        b = bars([(50, 50, 50, 50, 1e6)] * 2)
        a.iloc[1] = np.nan
        gappy = DataFeed({"A": a, "B": b})
        broker = SimulatedBroker()
        submit(broker, gappy, 1, BUY, 1, OrderType.MARKET_ON_CLOSE)
        assert run_bar(broker, gappy) == []
        assert broker.get(1).status is OrderStatus.EXPIRED
        feed = feed_of((100, 101, 98, 100, 1_000))
        capped = SimulatedBroker(max_participation=0.1)
        submit(capped, feed, 1, BUY, 80, OrderType.LIMIT, limit_price=99.0)
        submit(capped, feed, 2, BUY, 80, OrderType.MARKET_ON_CLOSE)
        assert run_bar(capped, feed) == [(1, 80, 99), (2, 20, 100)]  # 100 shares in total


# ---------------------------------------------------------------------- stop-limit
def test_stop_limit_revisited_limit_fills_only_under_the_optimistic_path() -> None:
    """Buy stop 102 / limit 101 on (100.6, 103, 100.5, 102). Path A triggers at 102 on
    the way up and comes back down through 101 (a fill worth +10); path B triggers on
    the final rally and never returns to 101 (worth 0). Worst: no fill."""
    feed = feed_of((100.6, 103, 100.5, 102, 1e6))
    fills = {}
    for policy in ("worst", "best"):
        broker = SimulatedBroker(intrabar=policy)  # type: ignore[arg-type]
        submit(
            broker,
            feed,
            1,
            BUY,
            10,
            OrderType.STOP_LIMIT,
            stop_price=102.0,
            limit_price=101.0,
            tif=GTC,
        )
        fills[policy] = run_bar(broker, feed)
        assert broker.get(1).triggered
    assert fills == {"worst": [], "best": [(1, 10, 101)]}


def test_invalid_policy() -> None:
    with pytest.raises(ConfigError, match="intrabar"):
        SimulatedBroker(intrabar="random")  # type: ignore[arg-type]
