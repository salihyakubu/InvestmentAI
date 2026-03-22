"""Order lifecycle management with state machine validation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

import structlog

from core.enums import OrderStatus

logger = structlog.get_logger(__name__)


class OrderError(Exception):
    """Raised on invalid order state transitions or operations."""


@dataclass
class Order:
    """Internal order representation."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    status: OrderStatus = OrderStatus.PENDING
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    external_id: str | None = None
    broker_name: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Fill:
    """A recorded fill against an order."""

    fill_id: str
    order_id: str
    price: Decimal
    quantity: Decimal
    commission: Decimal = Decimal("0")
    filled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OrderStateMachine:
    """Validates and enforces order state transitions.

    Only transitions defined in TRANSITIONS are permitted.
    """

    TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
        OrderStatus.PENDING: {
            OrderStatus.SUBMITTED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        },
        OrderStatus.SUBMITTED: {
            OrderStatus.PARTIAL_FILL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        },
        OrderStatus.PARTIAL_FILL: {
            OrderStatus.PARTIAL_FILL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        },
        OrderStatus.FILLED: set(),
        OrderStatus.CANCELLED: set(),
        OrderStatus.REJECTED: set(),
        OrderStatus.EXPIRED: set(),
    }

    @classmethod
    def transition(cls, order: Order, new_status: OrderStatus) -> Order:
        """Validate and apply a status transition on *order*.

        Raises OrderError if the transition is not allowed.
        """
        allowed = cls.TRANSITIONS.get(order.status, set())
        if new_status not in allowed:
            raise OrderError(
                f"Invalid transition: {order.status} -> {new_status} "
                f"(order {order.order_id})"
            )
        order.status = new_status
        order.updated_at = datetime.now(timezone.utc)
        logger.info(
            "order_status_transition",
            order_id=order.order_id,
            new_status=new_status,
        )
        return order


class OrderManager:
    """Manages order creation, status updates, and fill recording.

    Acts as the single source of truth for in-flight orders within the
    execution engine.
    """

    def __init__(self) -> None:
        # order_id -> Order
        self._orders: dict[str, Order] = {}
        # order_id -> list[Fill]
        self._fills: dict[str, list[Fill]] = {}

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        limit_price: Decimal | None = None,
        stop_price: Decimal | None = None,
    ) -> Order:
        """Create a new order in PENDING status."""
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
        )
        self._orders[order_id] = order
        self._fills[order_id] = []
        logger.info(
            "order_created",
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=str(quantity),
        )
        return order

    def update_status(self, order_id: str, status: OrderStatus) -> Order:
        """Transition an order to a new status."""
        order = self._get_order(order_id)
        return OrderStateMachine.transition(order, status)

    def record_fill(
        self,
        order_id: str,
        price: Decimal,
        quantity: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> Fill:
        """Record a fill and update the order's filled quantities."""
        order = self._get_order(order_id)

        fill_id = str(uuid.uuid4())
        fill = Fill(
            fill_id=fill_id,
            order_id=order_id,
            price=price,
            quantity=quantity,
            commission=commission,
        )
        self._fills[order_id].append(fill)

        # Recompute average fill price.
        total_cost = sum(f.price * f.quantity for f in self._fills[order_id])
        total_qty = sum(f.quantity for f in self._fills[order_id])
        order.filled_quantity = total_qty
        order.avg_fill_price = (total_cost / total_qty) if total_qty else None

        # Auto-transition to PARTIAL_FILL or FILLED.
        if total_qty >= order.quantity:
            OrderStateMachine.transition(order, OrderStatus.FILLED)
        elif order.status == OrderStatus.SUBMITTED:
            OrderStateMachine.transition(order, OrderStatus.PARTIAL_FILL)

        logger.info(
            "fill_recorded",
            order_id=order_id,
            fill_id=fill_id,
            price=str(price),
            quantity=str(quantity),
            total_filled=str(total_qty),
        )
        return fill

    def get_open_orders(self) -> list[Order]:
        """Return all orders that are not in a terminal state."""
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
        return [o for o in self._orders.values() if o.status not in terminal]

    def cancel_all_open(self) -> int:
        """Cancel all open orders. Returns count of orders cancelled."""
        open_orders = self.get_open_orders()
        count = 0
        for order in open_orders:
            try:
                OrderStateMachine.transition(order, OrderStatus.CANCELLED)
                count += 1
            except OrderError:
                logger.warning(
                    "cancel_all_skip",
                    order_id=order.order_id,
                    status=order.status,
                )
        logger.info("cancel_all_open_complete", cancelled=count)
        return count

    def get_order(self, order_id: str) -> Order | None:
        """Return an order by ID, or None if not found."""
        return self._orders.get(order_id)

    def _get_order(self, order_id: str) -> Order:
        """Return an order by ID or raise OrderError."""
        order = self._orders.get(order_id)
        if order is None:
            raise OrderError(f"Order not found: {order_id}")
        return order
