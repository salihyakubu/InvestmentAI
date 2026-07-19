"""Operator control endpoints -- the remote kill switch.

The execution engine's safety primitives (halt / resume / emergency-flatten)
are methods on a process inside the worker; without an authenticated remote
trigger they are unreachable exactly when they are needed most. These
endpoints publish a ``TradingControlEvent`` the execution engine consumes.
Admin role required.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.dependencies import get_current_user, get_event_bus
from core.events.base import EventBus
from core.events.order_events import TradingControlEvent
from core.events.streams import CONTROL

router = APIRouter()


class ControlRequest(BaseModel):
    action: Literal["halt", "resume", "flatten"]
    reason: str = ""


@router.post(
    "/control",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trading kill switch: halt, resume, or flatten",
)
async def trading_control(
    body: ControlRequest,
    user: dict[str, Any] = Depends(get_current_user),
    event_bus: EventBus = Depends(get_event_bus),
) -> dict[str, str]:
    """Publish an operator control command to the execution engine.

    ``halt`` stops new orders; ``resume`` lifts a halt; ``flatten`` cancels all
    open orders, closes every position, and halts. Asynchronous: 202 means the
    command was published -- confirm the effect via worker logs / positions.
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required for trading control",
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
