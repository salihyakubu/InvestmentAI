"""Audit log endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.audit import AuditLog
from api.dependencies import get_current_user, get_db

router = APIRouter(prefix="/audit")


@router.get(
    "/logs",
    summary="Query audit logs",
)
async def list_audit_logs(
    service: str | None = Query(default=None),
    action: str | None = Query(default=None),
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """Return audit log entries with optional filtering."""
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)

    if service is not None:
        stmt = stmt.where(AuditLog.service == service)
    if action is not None:
        stmt = stmt.where(AuditLog.action == action)
    if start is not None:
        stmt = stmt.where(AuditLog.timestamp >= start)
    if end is not None:
        stmt = stmt.where(AuditLog.timestamp <= end)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "id": str(row.id),
            "timestamp": row.timestamp.isoformat(),
            "service": row.service,
            "action": row.action,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "user_id": str(row.user_id) if row.user_id else None,
            "details": row.details,
        }
        for row in rows
    ]


@router.get(
    "/decision-trace/{order_id}",
    summary="Full decision trace for an order",
)
async def get_decision_trace(
    order_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict:
    """Return the full decision trace for a specific order.

    Includes signal generation, risk evaluation, and execution details.
    """
    stmt = (
        select(AuditLog)
        .where(
            AuditLog.entity_type == "order",
            AuditLog.entity_id == str(order_id),
        )
        .order_by(AuditLog.timestamp.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No decision trace found for order {order_id}",
        )

    return {
        "order_id": str(order_id),
        "trace": [
            {
                "timestamp": row.timestamp.isoformat(),
                "service": row.service,
                "action": row.action,
                "details": row.details,
                "decision_trace": row.decision_trace,
            }
            for row in rows
        ],
    }
