"""Risk management events."""

from __future__ import annotations

from typing import Any

from core.events.base import Event


class RiskApprovedEvent(Event):
    """Emitted when an order passes all pre-trade risk checks."""

    order_id: str


class RiskBreachedEvent(Event):
    """Emitted when a risk metric exceeds its configured limit."""

    rule_name: str
    current_value: float
    limit_value: float
    action: str  # e.g. 'warn', 'block', 'reduce'


class CircuitBreakerEvent(Event):
    """Emitted when the circuit breaker changes state."""

    state: str  # 'open', 'closed', 'half_open'
    reason: str


class RebalanceRequestEvent(Event):
    """Emitted to request portfolio rebalancing to target weights."""

    target_allocations: dict[str, float]
