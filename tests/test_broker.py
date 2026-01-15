"""Fill logic of the simulated broker, one order type at a time."""

from __future__ import annotations

import pandas as pd
import pytest

from backtester import (
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
from tests.conftest import Row, bars


def feed_of(*rows: Row) -> DataFeed:
    # bar 0 is the submission bar; the tested bars follow
    return DataFeed({"A": bars([(100, 100, 100, 100, 1e6), *rows])})


def submit(
    broker: SimulatedBroker,
    feed: DataFeed,
    side: Side,
    qty: float = 10,
    kind: OrderType = OrderType.MARKET,
    *,
    limit: float | None = None,
    stop: float | None = None,
    tif: TimeInForce = TimeInForce.DAY,
    order_id: int = 1,
    symbol: str = "A",
) -> None:
    broker.submit(
        OrderEvent(
            feed.index[0],
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=qty,
            order_type=kind,
            tif=tif,
            limit_price=limit,
            stop_price=stop,
        )
    )


def run_bar(broker: SimulatedBroker, feed: DataFeed, pos: int = 1) -> list[tuple[float, float]]:
    return [(f.quantity, f.price) for f in broker.process_bar(feed.view(pos))]


def test_market_order_fills_at_next_open_never_on_submission_bar() -> None:
    feed = feed_of((101, 103, 99, 102, 1e6))
    broker = SimulatedBroker()
    submit(broker, feed, Side.BUY)
    assert run_bar(broker, feed, pos=0) == []  # same bar as submission
    assert run_bar(broker, feed) == [(10, 101)]
    assert broker.get(1).status is OrderStatus.FILLED


@pytest.mark.parametrize(
    ("side", "limit", "bar", "expected"),
    [
        (Side.BUY, 99.0, (98, 100, 97, 99.5, 1e6), 98.0),  # gap below the limit: fill at open
        (Side.BUY, 99.0, (100, 101, 98, 100, 1e6), 99.0),  # range touches: fill at limit
        (Side.BUY, 99.0, (100, 101, 99.5, 100, 1e6), None),  # never reached
        (Side.SELL, 101.0, (102, 103, 101.5, 102, 1e6), 102.0),  # gap above: fill at open
        (Side.SELL, 101.0, (100, 101.5, 99, 100, 1e6), 101.0),
        (Side.SELL, 101.0, (100, 100.5, 99, 100, 1e6), None),
    ],
)
def test_limit_orders(side: Side, limit: float, bar: Row, expected: float | None) -> None:
    feed = feed_of(bar)
    broker = SimulatedBroker()
    submit(broker, feed, side, kind=OrderType.LIMIT, limit=limit)
    fills = run_bar(broker, feed)
    assert fills == ([] if expected is None else [(10, expected)])


def test_strict_limits_require_trading_through() -> None:
    feed = feed_of((100, 101, 99.0, 100, 1e6))
    for strict, expected in ((False, [(10, 99.0)]), (True, [])):
        broker = SimulatedBroker(strict_limits=strict)
        submit(broker, feed, Side.BUY, kind=OrderType.LIMIT, limit=99.0)
        assert run_bar(broker, feed) == expected


@pytest.mark.parametrize(
    ("side", "stop", "bar", "expected"),
    [
        (Side.BUY, 102.0, (104, 105, 103, 104, 1e6), 104.0),  # gaps through: fill at (worse) open
        (Side.BUY, 102.0, (100, 103, 99, 101, 1e6), 102.0),  # triggers intrabar: at stop
        (Side.BUY, 102.0, (100, 101, 99, 100, 1e6), None),
        (Side.SELL, 98.0, (95, 96, 94, 95, 1e6), 95.0),  # gap down through a stop-loss
        (Side.SELL, 98.0, (100, 101, 97, 99, 1e6), 98.0),
        (Side.SELL, 98.0, (100, 101, 99, 100, 1e6), None),
    ],
)
def test_stop_orders(side: Side, stop: float, bar: Row, expected: float | None) -> None:
    feed = feed_of(bar)
    broker = SimulatedBroker()
    submit(broker, feed, side, kind=OrderType.STOP, stop=stop)
    fills = run_bar(broker, feed)
    assert fills == ([] if expected is None else [(10, expected)])


class TestStopLimit:
    def fill(self, side: Side, stop: float, limit: float, *rows: Row) -> list[list[tuple]]:
        feed = feed_of(*rows)
        broker = SimulatedBroker()
        submit(
            broker,
            feed,
            side,
            kind=OrderType.STOP_LIMIT,
            stop=stop,
            limit=limit,
            tif=TimeInForce.GTC,
        )
        return [run_bar(broker, feed, pos=i) for i in range(1, len(rows) + 1)]

    def test_gap_through_stop_within_limit_fills_at_open(self) -> None:
        assert self.fill(Side.BUY, 102, 105, (103, 106, 102.5, 104, 1e6)) == [[(10, 103)]]

    def test_gap_through_stop_and_limit_waits_for_limit(self) -> None:
        # opens above the limit; later trades back down to it
        assert self.fill(Side.BUY, 102, 104, (106, 107, 103, 105, 1e6)) == [[(10, 104)]]

    def test_gap_through_both_and_limit_never_reached(self) -> None:
        assert self.fill(Side.BUY, 102, 104, (106, 107, 105, 106, 1e6)) == [[]]

    def test_intrabar_trigger_with_marketable_limit_fills_at_stop(self) -> None:
        assert self.fill(Side.SELL, 98, 97, (100, 101, 96, 99, 1e6)) == [[(10, 98)]]

    def test_intrabar_trigger_with_limit_beyond_stop_defers_to_later_bars(self) -> None:
        # buy stop 102 / limit 101: triggered mid-bar, but the path afterwards is unknown
        result = self.fill(
            Side.BUY,
            102,
            101,
            (100.6, 103, 100.5, 102, 1e6),  # triggers; the low may precede the trigger
            (102, 102.5, 100.8, 101.5, 1e6),  # now a working limit at 101: fills at 101
        )
        assert result == [[], [(10, 101)]]

    def test_not_triggered(self) -> None:
        assert self.fill(Side.BUY, 102, 104, (100, 101, 99, 100, 1e6)) == [[]]


def test_day_orders_expire_and_gtc_orders_persist() -> None:
    feed = feed_of((100, 101, 99.5, 100, 1e6), (100, 101, 98, 99, 1e6))
    broker = SimulatedBroker()
    submit(broker, feed, Side.BUY, kind=OrderType.LIMIT, limit=99, order_id=1)
    submit(broker, feed, Side.BUY, kind=OrderType.LIMIT, limit=99, tif=TimeInForce.GTC, order_id=2)
    assert run_bar(broker, feed, 1) == []
    assert broker.get(1).status is OrderStatus.EXPIRED
    assert broker.get(2).status is OrderStatus.NEW
    assert run_bar(broker, feed, 2) == [(10, 99)]
    assert broker.get(2).status is OrderStatus.FILLED
    assert broker.open_orders() == []


def test_participation_cap_produces_partial_fills() -> None:
    feed = feed_of((100, 101, 99, 100, 1_000), (100, 101, 99, 100, 1_000))
    broker = SimulatedBroker(max_participation=0.1)
    submit(broker, feed, Side.BUY, qty=150, tif=TimeInForce.GTC, order_id=1)
    submit(broker, feed, Side.SELL, qty=30, order_id=2)  # DAY: remainder expires
    assert run_bar(broker, feed, 1) == [(100, 100)]  # cap is shared per symbol and bar
    order = broker.get(1)
    assert order.status is OrderStatus.PARTIALLY_FILLED
    assert order.remaining == 50
    assert broker.get(2).status is OrderStatus.EXPIRED
    assert broker.get(2).filled_quantity == 0
    assert run_bar(broker, feed, 2) == [(50, 100)]
    assert order.status is OrderStatus.FILLED
    assert order.avg_fill_price == 100


def test_zero_volume_bar_cannot_fill_with_a_cap() -> None:
    feed = feed_of((100, 101, 99, 100, 0))
    broker = SimulatedBroker(max_participation=0.5)
    submit(broker, feed, Side.BUY)
    assert run_bar(broker, feed) == []


def test_slippage_is_adverse_and_limits_are_respected() -> None:
    feed = feed_of((100, 101, 98, 100, 1e6))
    broker = SimulatedBroker(slippage=FixedBpsSlippage(50))
    submit(broker, feed, Side.BUY, order_id=1)
    submit(broker, feed, Side.SELL, order_id=2)
    submit(broker, feed, Side.BUY, kind=OrderType.LIMIT, limit=99, order_id=3)
    fills = {f.order_id: f for f in broker.process_bar(feed.view(1))}
    assert fills[1].price == pytest.approx(100.5)
    assert fills[2].price == pytest.approx(99.5)
    assert fills[1].slippage == pytest.approx(5.0)
    assert fills[3].price == 99  # slippage cannot push a limit fill past its limit


def test_missing_bar_means_no_fill() -> None:
    a = bars([(100, 100, 100, 100, 1e6)] * 2)
    b = bars([(50, 50, 50, 50, 1e6)] * 2)
    b.iloc[1] = float("nan")
    feed = DataFeed({"A": a, "B": b})
    broker = SimulatedBroker()
    submit(broker, feed, Side.BUY, symbol="B")
    assert run_bar(broker, feed) == []
    assert broker.get(1).status is OrderStatus.EXPIRED


def test_cancel_and_duplicates() -> None:
    feed = feed_of((100, 101, 99, 100, 1e6))
    broker = SimulatedBroker()
    submit(broker, feed, Side.BUY, order_id=1)
    submit(broker, feed, Side.BUY, order_id=2)
    ts = pd.Timestamp(feed.index[0])
    assert broker.cancel(1, ts)
    assert not broker.cancel(1, ts)
    assert [o.id for o in broker.cancel_all(ts, symbol="A")] == [2]
    assert run_bar(broker, feed) == []
    with pytest.raises(OrderError):
        submit(broker, feed, Side.BUY, order_id=2)
