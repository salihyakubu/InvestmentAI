"""Operator control endpoints -- the remote kill switch.

The execution engine's safety primitives (halt / resume / emergency-flatten)
are methods on a process inside the worker; without an authenticated remote
trigger they are unreachable exactly when they are needed most. These
endpoints publish a ``TradingControlEvent`` the execution engine consumes.
Admin role required.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from api.dependencies import client_ip, get_audit_logger, get_event_bus, require_role
from core.events.base import EventBus
from core.events.order_events import TradingControlEvent
from core.events.streams import CONTROL
from services.monitoring.audit import AuditLogger

router = APIRouter()


class ControlRequest(BaseModel):
    action: Literal["halt", "resume", "flatten", "retrain"]
    reason: str = ""


@router.post(
    "/control",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trading kill switch: halt, resume, or flatten",
)
async def trading_control(
    body: ControlRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_role("admin")),
    event_bus: EventBus = Depends(get_event_bus),
    audit: AuditLogger = Depends(get_audit_logger),
) -> dict[str, str]:
    """Publish an operator control command to the execution engine.

    ``halt`` stops new orders; ``resume`` lifts a halt; ``flatten`` cancels all
    open orders, closes every position, and halts. ``retrain`` triggers an
    immediate evaluation/retrain cycle in the worker's continuous-learning
    service. Asynchronous: 202 means the command was published -- confirm the
    effect via worker logs / positions.
    """
    # The durable, attributable record comes first: who, from where, why.
    await audit.log_action(
        service="api",
        action=f"control.{body.action}",
        entity_type="trading_control",
        entity_id=body.action,
        details={"reason": body.reason},
        user_id=user.get("user_id"),
        ip_address=client_ip(request),
    )
    await event_bus.publish(
        CONTROL,
        TradingControlEvent(
            action=body.action,
            reason=body.reason or f"manual via API by {user.get('user_id')}",
            source_service="api",
        ),
    )
    return {"status": "accepted", "action": body.action}
