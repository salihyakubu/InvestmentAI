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
    symbol: str = ""
    side: str = ""  # OrderSide value
    client_order_id: str = ""  # originating DB order id, if any


class OrderRejectedEvent(Event):
    """Emitted when an order is rejected by risk or the broker."""

    order_id: str
    reason: str
    client_order_id: str = ""  # originating DB order id, if any


class OrderCancelledEvent(Event):
    """Emitted when an order is cancelled."""

    order_id: str
    reason: str = ""


class OrderIntentEvent(Event):
    """A manual order request published by the API for the execution worker.

    Carries the full order so execution can create and submit it; ``order_id``
    is the DB order id, used as a correlation handle.
    """

    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    order_id: str = ""
    limit_price: float | None = None
    stop_price: float | None = None


class TradingControlEvent(Event):
    """An operator control command for the execution engine (the kill switch).

    Actions: ``halt`` (stop accepting new orders), ``resume`` (lift a halt),
    ``flatten`` (cancel all open orders, close every position, and halt).
    Published by the admin API; consumed by the execution engine.
    """

    action: str
    reason: str = ""
