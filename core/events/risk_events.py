"""Risk management events."""

from __future__ import annotations

from pydantic import Field

from core.events.base import Event


class RiskApprovedEvent(Event):
    """Emitted when an order passes all pre-trade risk checks.

    Carries the full, sized order intent so the execution engine can create and
    submit the order directly (the risk service runs in a separate process with
    its own order state, so an order id alone is not resolvable downstream).
    """

    order_id: str = ""          # correlation id, not the execution order id
    symbol: str = ""
    side: str = ""              # OrderSide value
    order_type: str = "market"  # OrderType value
    quantity: float = 0.0
    limit_price: float | None = None
    stop_price: float | None = None
    reference_price: float | None = None  # price used to size the quantity


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
    # symbol -> reference price, supplied by the producer (which has market
    # data) so the risk manager can size weight deltas into share quantities.
    reference_prices: dict[str, float] = Field(default_factory=dict)
