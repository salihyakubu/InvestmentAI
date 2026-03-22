"""Portfolio endpoints."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.portfolio import PortfolioSnapshot
from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse
from api.schemas.portfolio import AllocationSchema, PortfolioSummary

router = APIRouter(prefix="/portfolio")


@router.get(
    "/summary",
    response_model=PortfolioSummary,
    summary="Current portfolio state",
)
async def get_portfolio_summary(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> PortfolioSummary:
    """Return the latest portfolio snapshot as a summary."""
    stmt = (
        select(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    snapshot = result.scalar_one_or_none()

    if snapshot is None:
        return PortfolioSummary(
            total_equity=Decimal("0"),
            cash=Decimal("0"),
            positions_value=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            daily_return_pct=None,
        )

    return PortfolioSummary(
        total_equity=snapshot.total_equity,
        cash=snapshot.cash,
        positions_value=snapshot.positions_value,
        unrealized_pnl=snapshot.unrealized_pnl,
        realized_pnl=snapshot.realized_pnl,
        daily_return_pct=snapshot.daily_return_pct,
    )


@router.get(
    "/allocations",
    response_model=list[AllocationSchema],
    summary="Current portfolio allocations",
)
async def get_allocations(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[AllocationSchema]:
    """Return the current asset allocations from the latest snapshot."""
    stmt = (
        select(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    snapshot = result.scalar_one_or_none()

    if snapshot is None or not snapshot.allocations:
        return []

    allocations: list[AllocationSchema] = []
    for symbol, data in snapshot.allocations.items():
        allocations.append(
            AllocationSchema(
                symbol=symbol,
                weight=data.get("weight", 0.0),
                value=Decimal(str(data.get("value", 0))),
            )
        )
    return allocations


@router.get(
    "/snapshots",
    summary="Historical portfolio snapshots",
)
async def get_snapshots(
    start: datetime | None = Query(default=None),
    end: datetime | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> list[dict]:
    """Return historical portfolio snapshots."""
    stmt = (
        select(PortfolioSnapshot)
        .order_by(PortfolioSnapshot.time.desc())
        .limit(limit)
    )
    if start is not None:
        stmt = stmt.where(PortfolioSnapshot.time >= start)
    if end is not None:
        stmt = stmt.where(PortfolioSnapshot.time <= end)

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        {
            "time": row.time.isoformat(),
            "total_equity": float(row.total_equity),
            "cash": float(row.cash),
            "positions_value": float(row.positions_value),
            "unrealized_pnl": float(row.unrealized_pnl),
            "realized_pnl": float(row.realized_pnl),
            "daily_return_pct": row.daily_return_pct,
            "position_count": row.position_count,
        }
        for row in rows
    ]


@router.post(
    "/rebalance",
    response_model=SuccessResponse,
    summary="Trigger manual rebalance",
)
async def trigger_rebalance(
    _user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Trigger a manual portfolio rebalance.

    This enqueues a rebalance task; the actual rebalancing is handled
    asynchronously by the portfolio service.
    """
    # TODO: Publish rebalance event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Rebalance service not yet implemented",
    )
