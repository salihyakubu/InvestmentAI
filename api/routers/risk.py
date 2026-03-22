"""Risk management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.risk import RiskMetric
from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse
from api.schemas.risk import RiskMetricsSchema

router = APIRouter(prefix="/risk")


@router.get(
    "/metrics/latest",
    response_model=RiskMetricsSchema,
    summary="Current risk metrics",
)
async def get_latest_risk_metrics(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> RiskMetricsSchema:
    """Return the most recent risk metrics snapshot."""
    stmt = (
        select(RiskMetric)
        .order_by(RiskMetric.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        return RiskMetricsSchema()

    return RiskMetricsSchema(
        var_95=row.var_95,
        var_99=row.var_99,
        cvar_95=row.cvar_95,
        max_drawdown=row.max_drawdown,
        current_drawdown=row.current_drawdown,
        circuit_breaker_active=row.circuit_breaker_active,
    )


@router.get(
    "/var",
    summary="Current Value at Risk",
)
async def get_var(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict:
    """Return the current VaR figures."""
    stmt = (
        select(RiskMetric)
        .order_by(RiskMetric.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    if row is None:
        return {"var_95": None, "var_99": None, "cvar_95": None}

    return {
        "var_95": row.var_95,
        "var_99": row.var_99,
        "cvar_95": row.cvar_95,
    }


@router.get(
    "/circuit-breaker/status",
    summary="Circuit breaker state",
)
async def get_circuit_breaker_status(
    db: AsyncSession = Depends(get_db),
    _user: dict = Depends(get_current_user),
) -> dict:
    """Return whether the circuit breaker is currently active."""
    stmt = (
        select(RiskMetric)
        .order_by(RiskMetric.time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    row = result.scalar_one_or_none()

    return {
        "circuit_breaker_active": row.circuit_breaker_active if row else False,
        "current_drawdown": row.current_drawdown if row else None,
    }


@router.post(
    "/circuit-breaker/reset",
    response_model=SuccessResponse,
    summary="Manually reset circuit breaker",
)
async def reset_circuit_breaker(
    _user: dict = Depends(get_current_user),
) -> SuccessResponse:
    """Manually reset the circuit breaker.

    Requires admin role. Will be connected to the risk service.
    """
    # TODO: Publish circuit-breaker reset event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Circuit breaker reset not yet implemented",
    )
