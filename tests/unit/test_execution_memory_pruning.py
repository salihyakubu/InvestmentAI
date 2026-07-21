"""Execution-stack memory must stay bounded over a long soak.

Covers the per-order fill index and 48h terminal-order prune in PaperBroker,
the matching prune in OrderManager, and the removal of FillTracker's
append-only per-order fill history: all three grew for the process lifetime.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from core.enums import OrderStatus
from services.execution.brokers.base import BrokerOrder
from services.execution.brokers.paper_broker import PaperBroker
from services.execution.fill_tracker import FillTracker
from services.execution.order_manager import OrderManager


def _market_order(symbol: str, qty: str = "10", side: str = "buy") -> BrokerOrder:
    return BrokerOrder(
        external_id="",
        symbol=symbol,
        side=side,
        order_type="market",
        quantity=Decimal(qty),
    )


def _limit_order(symbol: str, limit: str, qty: str = "5") -> BrokerOrder:
    return BrokerOrder(
        external_id="",
        symbol=symbol,
        side="buy",
        order_type="limit",
        quantity=Decimal(qty),
        limit_price=Decimal(limit),
    )


# ----------------------------------------------------------------------
# PaperBroker: fills index correctness
# ----------------------------------------------------------------------


async def test_paper_broker_fill_index_matches_full_scan() -> None:
    broker = PaperBroker()
    broker.update_price("AAPL", Decimal("100"))
    broker.update_price("MSFT", Decimal("50"))

    ext_ids = [
        await broker.submit_order(_market_order(sym, qty=qty))
        for sym, qty in (("AAPL", "10"), ("MSFT", "4"), ("AAPL", "6"))
    ]

    all_fills = broker.fills
    assert len(all_fills) == 3

    for ext_id in ext_ids:
        # Old behavior: linear scan over every fill ever recorded.
        scanned = [f for f in all_fills if f.order_external_id == ext_id]
        scanned_qty = sum((f.quantity for f in scanned), Decimal("0"))
        scanned_avg = sum((f.price * f.quantity for f in scanned), Decimal("0")) / scanned_qty

        status = await broker.get_order_status(ext_id)
        assert status["status"] == "filled"
        assert Decimal(status["filled_qty"]) == scanned_qty
        assert Decimal(status["filled_avg_price"]) == scanned_avg


async def test_paper_broker_limit_fill_lands_in_index() -> None:
    broker = PaperBroker()
    broker.update_price("AAPL", Decimal("100"))

    ext_id = await broker.submit_order(_limit_order("AAPL", limit="95"))
    status = await broker.get_order_status(ext_id)
    assert status["status"] == "submitted"
    assert status["filled_qty"] == "0"

    broker.update_price("AAPL", Decimal("94"))
    status = await broker.get_order_status(ext_id)
    assert status["status"] == "filled"
    assert status["filled_qty"] == "5"
    assert Decimal(status["filled_avg_price"]) == Decimal("94.00")


# ----------------------------------------------------------------------
# PaperBroker: terminal-order pruning
# ----------------------------------------------------------------------


async def test_paper_broker_submit_prunes_only_old_terminal_orders() -> None:
    broker = PaperBroker()
    broker.update_price("AAPL", Decimal("100"))

    old_filled = await broker.submit_order(_market_order("AAPL"))
    recent_filled = await broker.submit_order(_market_order("AAPL"))
    open_limit = await broker.submit_order(_limit_order("AAPL", limit="10"))

    broker._terminal_at[old_filled] = datetime.now(UTC) - timedelta(hours=49)

    # Prune runs opportunistically on the next submission.
    await broker.submit_order(_market_order("AAPL", qty="1"))

    assert old_filled not in broker._orders
    assert old_filled not in broker._order_statuses
    assert old_filled not in broker._fills_by_order
    assert (await broker.get_order_status(old_filled))["status"] == "unknown"

    assert (await broker.get_order_status(recent_filled))["status"] == "filled"
    assert (await broker.get_order_status(open_limit))["status"] == "submitted"
    assert open_limit in broker.open_orders


async def test_paper_broker_prune_leaves_account_state_untouched() -> None:
    broker = PaperBroker()
    broker.update_price("AAPL", Decimal("100"))

    ext_id = await broker.submit_order(_market_order("AAPL"))
    account_before = await broker.get_account()
    positions_before = await broker.get_positions()

    broker._terminal_at[ext_id] = datetime.now(UTC) - timedelta(hours=49)
    broker._prune()

    assert ext_id not in broker._fills_by_order
    assert await broker.get_account() == account_before
    assert await broker.get_positions() == positions_before


async def test_paper_broker_cancelled_and_rejected_orders_are_prunable() -> None:
    broker = PaperBroker()

    # Rejected: market order with no known price and no limit fallback.
    rejected = await broker.submit_order(_market_order("NOPX"))
    assert (await broker.get_order_status(rejected))["status"] == "rejected"

    broker.update_price("AAPL", Decimal("100"))
    cancelled = await broker.submit_order(_limit_order("AAPL", limit="10"))
    assert await broker.cancel_order(cancelled)

    for ext_id in (rejected, cancelled):
        broker._terminal_at[ext_id] = datetime.now(UTC) - timedelta(hours=49)
    broker._prune()

    assert rejected not in broker._orders
    assert cancelled not in broker._orders
    assert not broker._terminal_at


# ----------------------------------------------------------------------
# OrderManager: terminal-order pruning
# ----------------------------------------------------------------------


def test_order_manager_create_prunes_only_old_terminal_orders() -> None:
    mgr = OrderManager()

    old_filled = mgr.create_order("AAPL", "buy", "market", Decimal("10"))
    mgr.update_status(old_filled.order_id, OrderStatus.SUBMITTED)
    mgr.record_fill(old_filled.order_id, Decimal("100"), Decimal("10"))
    assert old_filled.status == OrderStatus.FILLED
    old_filled.updated_at = datetime.now(UTC) - timedelta(hours=49)

    stale_open = mgr.create_order("MSFT", "buy", "limit", Decimal("5"), limit_price=Decimal("50"))
    mgr.update_status(stale_open.order_id, OrderStatus.SUBMITTED)
    # Older than the window but not terminal: must survive.
    stale_open.updated_at = datetime.now(UTC) - timedelta(hours=72)

    recent_cancelled = mgr.create_order("GOOG", "sell", "market", Decimal("1"))
    mgr.update_status(recent_cancelled.order_id, OrderStatus.CANCELLED)

    # Prune runs opportunistically on the next creation.
    mgr.create_order("TSLA", "buy", "market", Decimal("1"))

    assert mgr.get_order(old_filled.order_id) is None
    assert old_filled.order_id not in mgr._fills
    assert mgr.get_order(stale_open.order_id) is stale_open
    assert mgr.get_order(recent_cancelled.order_id) is recent_cancelled
    assert stale_open in mgr.get_open_orders()


# ----------------------------------------------------------------------
# FillTracker: per-order fill history removed
# ----------------------------------------------------------------------


def test_fill_tracker_position_math_without_per_order_history() -> None:
    tracker = FillTracker()

    pos = tracker.record_fill("order-1", "AAPL", "buy", Decimal("100"), Decimal("10"))
    tracker.record_fill("order-2", "AAPL", "sell", Decimal("110"), Decimal("4"))

    assert pos.quantity == Decimal("6")
    assert tracker.get_position("AAPL") is pos
    # The append-only per-order history (read by nothing) is gone.
    assert not hasattr(tracker, "_fills_by_order")
