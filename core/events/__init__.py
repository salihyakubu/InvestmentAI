"""Event system for inter-service communication."""

from core.events.base import Event, EventBus, InProcessEventBus
from core.events.market_events import BarCloseEvent, PriceUpdateEvent
from core.events.signal_events import FeaturesReadyEvent, PredictionReadyEvent
from core.events.order_events import (
    OrderCancelledEvent,
    OrderCreatedEvent,
    OrderFilledEvent,
    OrderRejectedEvent,
)
from core.events.risk_events import (
    CircuitBreakerEvent,
    RebalanceRequestEvent,
    RiskApprovedEvent,
    RiskBreachedEvent,
)
from core.events.system_events import (
    DriftDetectedEvent,
    LiquidationTriggeredEvent,
    ModelRetrainedEvent,
)

__all__ = [
    # base
    "Event",
    "EventBus",
    "InProcessEventBus",
    # market
    "BarCloseEvent",
    "PriceUpdateEvent",
    # signal
    "FeaturesReadyEvent",
    "PredictionReadyEvent",
    # order
    "OrderCreatedEvent",
    "OrderFilledEvent",
    "OrderRejectedEvent",
    "OrderCancelledEvent",
    # risk
    "RiskApprovedEvent",
    "RiskBreachedEvent",
    "CircuitBreakerEvent",
    "RebalanceRequestEvent",
    # system
    "ModelRetrainedEvent",
    "DriftDetectedEvent",
    "LiquidationTriggeredEvent",
]
