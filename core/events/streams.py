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

# Risk -> (monitoring): a breached risk limit.
RISK_BREACHED = "risk.breached"

# Execution -> (risk, monitoring): order lifecycle events
# (created / filled / rejected / cancelled) all flow on this stream.
ORDERS = "orders"

# API -> Execution: a manual order request for the worker to execute.
ORDER_INTENTS = "order.intents"

# Prediction -> Portfolio: model predictions ready for optimisation.
PREDICTIONS = "predictions"
