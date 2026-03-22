"""Position endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.positions import Position
from api.dependencies import get_current_user, get_db
from api.schemas.portfolio import PositionSchema

router = APIRouter(prefix="/positions")


@router.get(
    "",
    response_model=list[PositionSchema],
    summary="All open positions",
)
async def list_open_positions(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[PositionSchema]:
    """Return all currently open positions."""
    stmt = (
        select(Position)
        .where(Position.closed_at.is_(None))
        .order_by(Position.opened_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        PositionSchema(
            symbol=p.symbol,
            side=p.side,
            quantity=p.quantity,
            avg_entry=p.avg_entry_price,
            current_price=p.current_price,
            unrealized_pnl=p.unrealized_pnl,
        )
        for p in rows
    ]


@router.get(
    "/closed",
    response_model=list[PositionSchema],
    summary="Closed positions",
)
async def list_closed_positions(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[PositionSchema]:
    """Return recently closed positions."""
    stmt = (
        select(Position)
        .where(Position.closed_at.is_not(None))
        .order_by(Position.closed_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        PositionSchema(
            symbol=p.symbol,
            side=p.side,
            quantity=p.quantity,
            avg_entry=p.avg_entry_price,
            current_price=p.current_price,
            unrealized_pnl=p.unrealized_pnl,
        )
        for p in rows
    ]


@router.get(
    "/{position_id}",
    response_model=PositionSchema,
    summary="Position detail",
)
async def get_position(
    position_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> PositionSchema:
    """Return a single position by ID."""
    stmt = select(Position).where(Position.id == position_id)
    result = await db.execute(stmt)
    p = result.scalar_one_or_none()

    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Position {position_id} not found",
        )

    return PositionSchema(
        symbol=p.symbol,
        side=p.side,
        quantity=p.quantity,
        avg_entry=p.avg_entry_price,
        current_price=p.current_price,
        unrealized_pnl=p.unrealized_pnl,
    )
