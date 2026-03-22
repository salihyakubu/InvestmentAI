"""Unit tests for the order lifecycle state machine."""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.enums import OrderStatus
from services.execution.order_manager import (
    Order,
    OrderError,
    OrderManager,
    OrderStateMachine,
)


@pytest.fixture
def pending_order() -> Order:
    """Create a fresh order in PENDING status."""
    return Order(
        order_id="test-order-1",
        symbol="AAPL",
        side="buy",
        order_type="market",
        quantity=Decimal("10"),
    )


class TestValidTransitions:
    """Verify that allowed state transitions succeed."""

    def test_valid_transition_pending_to_submitted(
        self, pending_order: Order
    ) -> None:
        """PENDING -> SUBMITTED is a valid transition."""
        result = OrderStateMachine.transition(pending_order, OrderStatus.SUBMITTED)
        assert result.status == OrderStatus.SUBMITTED

    def test_valid_transition_submitted_to_filled(
        self, pending_order: Order
    ) -> None:
        """SUBMITTED -> FILLED is a valid transition."""
        OrderStateMachine.transition(pending_order, OrderStatus.SUBMITTED)
        result = OrderStateMachine.transition(pending_order, OrderStatus.FILLED)
        assert result.status == OrderStatus.FILLED


class TestInvalidTransitions:
    """Verify that disallowed state transitions raise OrderError."""

    def test_invalid_transition_filled_to_pending(
        self, pending_order: Order
    ) -> None:
        """FILLED -> PENDING is not allowed (terminal state)."""
        OrderStateMachine.transition(pending_order, OrderStatus.SUBMITTED)
        OrderStateMachine.transition(pending_order, OrderStatus.FILLED)

        with pytest.raises(OrderError, match="Invalid transition"):
            OrderStateMachine.transition(pending_order, OrderStatus.PENDING)


class TestPartialFillFlow:
    """Verify partial fill transitions."""

    def test_partial_fill_to_filled(self, pending_order: Order) -> None:
        """PARTIAL_FILL -> FILLED is valid when all quantity is filled."""
        OrderStateMachine.transition(pending_order, OrderStatus.SUBMITTED)
        OrderStateMachine.transition(pending_order, OrderStatus.PARTIAL_FILL)
        result = OrderStateMachine.transition(pending_order, OrderStatus.FILLED)
        assert result.status == OrderStatus.FILLED


class TestCancelFlow:
    """Verify order cancellation."""

    def test_cancel_from_submitted(self, pending_order: Order) -> None:
        """An order in SUBMITTED status can be cancelled."""
        OrderStateMachine.transition(pending_order, OrderStatus.SUBMITTED)
        result = OrderStateMachine.transition(pending_order, OrderStatus.CANCELLED)
        assert result.status == OrderStatus.CANCELLED


class TestOrderManagerFills:
    """Verify the OrderManager fill recording logic."""

    def test_record_fill_auto_transitions(self) -> None:
        """Recording a fill that completes the order should
        auto-transition to FILLED."""
        mgr = OrderManager()
        order = mgr.create_order(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=Decimal("10"),
        )
        mgr.update_status(order.order_id, OrderStatus.SUBMITTED)

        fill = mgr.record_fill(
            order.order_id,
            price=Decimal("150.00"),
            quantity=Decimal("10"),
        )

        assert fill.quantity == Decimal("10")
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == Decimal("150.00")
