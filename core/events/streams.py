"""Canonical event-bus stream names.

Single source of truth for the topic strings used with the event bus, so
producers and consumers cannot drift apart (e.g. ``risk_approved`` vs
``risk.approved``). New services should import these constants rather than
hard-coding stream names.
"""

from __future__ import annotations

# Portfolio -> Risk: request to rebalance to target weights.
REBALANCE = "rebalance"

# Risk -> Execution: a sized order that passed all pre-trade checks.
RISK_APPROVED = "risk.approved"

# Risk -> (liquidation, monitoring): a breached risk limit, and every circuit-
# breaker state transition (CircuitBreakerEvent). Until 2026-11-02 the
# liquidation manager subscribed to "risk.breaches" and monitoring to "risk",
# so no breach ever reached either consumer; import the constant.
RISK_BREACHED = "risk.breached"

# Liquidation -> Execution: a position that must be closed now (stop hit,
# emergency). Consumed by the execution engine through the tracked order path.
LIQUIDATION = "liquidation.triggered"

# Execution -> (risk, monitoring): order lifecycle events
# (created / filled / rejected / cancelled) all flow on this stream.
ORDERS = "orders"

# API -> Execution: a manual order request for the worker to execute.
ORDER_INTENTS = "order.intents"

# Prediction -> Portfolio: model predictions ready for optimisation. This is
# the stream the prediction service actually publishes on.
PREDICTIONS_READY = "predictions.ready"

# Legacy prediction stream name; superseded by PREDICTIONS_READY. Kept so any
# out-of-tree consumer groups keep resolving -- do not point new code here.
PREDICTIONS = "predictions"

# API (admin) -> Execution: operator control commands (halt / resume / flatten).
# This is the remote kill switch -- without it the execution engine's safety
# primitives are unreachable on a deployed system.
CONTROL = "control"

# Continuous learning -> (prediction, monitoring): platform-level system events
# (ModelRetrainedEvent / DriftDetectedEvent). The prediction service listens
# here to hot-reload newly promoted model versions without a restart.
SYSTEM = "system"
