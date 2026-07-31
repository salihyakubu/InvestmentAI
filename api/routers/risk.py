"""Risk management endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from api.schemas.common import SuccessResponse
from api.schemas.risk import RiskMetricsSchema
from core.models.risk import RiskMetric

router = APIRouter(prefix="/risk")


@router.get(
    "/metrics/latest",
    response_model=RiskMetricsSchema,
    summary="Current risk metrics",
)
async def get_latest_risk_metrics(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
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
        # No writer has published risk state: report nothing rather than a
        # zeroed, all-clear snapshot the UI would render as measured safety.
        return RiskMetricsSchema(reported=False)

    details = row.details or {}
    failed_rules = details.get("rules_failed") or []
    return RiskMetricsSchema(
        var_95=row.var_95,
        var_99=row.var_99,
        cvar_95=row.cvar_95,
        cvar_99=row.cvar_99,
        max_drawdown=row.max_drawdown,
        current_drawdown=row.current_drawdown,
        volatility=details.get("volatility"),
        correlation_max=row.correlation_max,
        concentration_max=row.concentration_max,
        daily_pnl_pct=details.get("daily_pnl_pct"),
        circuit_breaker_active=row.circuit_breaker_active,
        circuit_breaker_state=details.get("circuit_breaker_state"),
        circuit_breaker_reason=(
            f"failed rules: {', '.join(failed_rules)}" if failed_rules else None
        ),
        reported=True,
    )


@router.get(
    "/var",
    summary="Current Value at Risk",
)
async def get_var(
    db: AsyncSession = Depends(get_db),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
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
    _user: dict[str, Any] = Depends(get_current_user),
) -> SuccessResponse:
    """Manually reset the circuit breaker.

    Requires admin role. Will be connected to the risk service.
    """
    # TODO: Publish circuit-breaker reset event via EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Circuit breaker reset not yet implemented",
    )
