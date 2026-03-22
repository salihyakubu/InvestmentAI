"""Backtesting endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_current_user
from api.schemas.common import SuccessResponse

router = APIRouter(prefix="/backtest")


class BacktestRequest(BaseModel):
    """Parameters for a backtest run."""

    strategy_name: str
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=100.0, gt=0)
    parameters: dict | None = None


class BacktestStatusResponse(BaseModel):
    """Status of a running or completed backtest."""

    backtest_id: str
    status: str  # queued, running, completed, failed
    progress_pct: float | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@router.post(
    "/run",
    response_model=BacktestStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a backtest",
)
async def start_backtest(
    params: BacktestRequest,
    _user: dict = Depends(get_current_user),
) -> BacktestStatusResponse:
    """Submit a new backtest run.

    The backtest executes asynchronously. Use the returned ID to poll status.
    """
    # TODO: Enqueue backtest job via task queue / EventBus
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Backtesting engine not yet implemented",
    )


@router.get(
    "/{backtest_id}/status",
    response_model=BacktestStatusResponse,
    summary="Check backtest status",
)
async def get_backtest_status(
    backtest_id: uuid.UUID,
    _user: dict = Depends(get_current_user),
) -> BacktestStatusResponse:
    """Return the current status of a backtest run."""
    # TODO: Look up backtest job status
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Backtesting engine not yet implemented",
    )


@router.get(
    "/{backtest_id}/results",
    summary="Get backtest results",
)
async def get_backtest_results(
    backtest_id: uuid.UUID,
    _user: dict = Depends(get_current_user),
) -> dict:
    """Return the results of a completed backtest.

    Includes equity curve, trade log, and performance statistics.
    """
    # TODO: Retrieve backtest results from storage
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Backtesting engine not yet implemented",
    )
