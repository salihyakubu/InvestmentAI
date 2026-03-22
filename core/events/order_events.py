"""Order lifecycle events."""

from __future__ import annotations

from core.events.base import Event


class OrderCreatedEvent(Event):
    """Emitted when a new order is created and submitted to risk checks."""

    order_id: str
    symbol: str
    side: str       # OrderSide value
    order_type: str  # OrderType value
    quantity: float


class OrderFilledEvent(Event):
    """Emitted when an order (or part of an order) is filled."""

    order_id: str
    fill_price: float
    fill_quantity: float
    commission: float = 0.0


class OrderRejectedEvent(Event):
    """Emitted when an order is rejected by risk or the broker."""

    order_id: str
    reason: str


class OrderCancelledEvent(Event):
    """Emitted when an order is cancelled."""

    order_id: str
    reason: str = ""
