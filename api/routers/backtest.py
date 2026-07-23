"""Backtest endpoints -- job-based execution behind the dashboard's page.

POST /backtest/run creates a job row and spawns the runner as a background
asyncio task in this process; the UI polls GET /backtest/status/{id} and
fetches GET /backtest/results/{id} on completion. The engine is the Stage-2
edge harness (net-of-cost, non-overlapping holds, stability gates) -- see
``services.backtesting.edge``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from core.models.backtests import BacktestJob
from services.backtesting.runner import STALE_RUNNING_AFTER_S, execute_job

router = APIRouter(prefix="/backtest")

_MAX_SYMBOLS = 10
_MAX_RANGE_DAYS = 15 * 365

# Keep strong references to spawned jobs so they are not garbage-collected
# mid-run (asyncio only holds weak refs to tasks).
_running_tasks: set[asyncio.Task[None]] = set()


class BacktestRunRequest(BaseModel):
    start_date: str
    end_date: str
    symbols: list[str] = Field(min_length=1, max_length=_MAX_SYMBOLS)
    strategy: str = "edge_harness"
    initial_capital: float = Field(default=10_000.0, gt=0)
    commission: float = Field(default=0.0005, ge=0, le=0.05)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        datetime.fromisoformat(v)  # raises on garbage
        return v


def _job_status(job: BacktestJob) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


async def _get_job(db: AsyncSession, job_id: uuid.UUID) -> BacktestJob:
    job = (
        await db.execute(select(BacktestJob).where(BacktestJob.id == job_id))
    ).scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Backtest {job_id} not found"
        )
    # A job orphaned by an API restart stays "running" forever; surface that
    # honestly instead of letting the UI poll into eternity.
    if job.status == "running" and job.started_at is not None:
        started = job.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if datetime.now(UTC) - started > timedelta(seconds=STALE_RUNNING_AFTER_S):
            job.status = "failed"
            job.error = "job orphaned (API restarted mid-run); re-run the backtest"
            job.finished_at = datetime.now(UTC)
            await db.commit()
    return job


@router.post("/run", status_code=status.HTTP_202_ACCEPTED, summary="Start a backtest job")
async def run_backtest_job(
    body: BacktestRunRequest,
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    start = datetime.fromisoformat(body.start_date)
    end = datetime.fromisoformat(body.end_date)
    if end <= start:
        raise HTTPException(status_code=422, detail="end_date must be after start_date")
    if (end - start).days > _MAX_RANGE_DAYS:
        raise HTTPException(status_code=422, detail="date range too large (15y max)")

    config = body.model_dump()
    job = BacktestJob(status="queued", config=config)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from core.models.base import get_async_session_factory

    task = asyncio.create_task(
        execute_job(job.id, config, get_async_session_factory()),
        name=f"backtest-{job.id}",
    )
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)

    return {"id": str(job.id), "status": "queued"}


@router.get("/status/{job_id}", summary="Backtest job status")
async def backtest_status(
    job_id: uuid.UUID,
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    return _job_status(await _get_job(db, job_id))


@router.get("/results/{job_id}", summary="Backtest results")
async def backtest_results(
    job_id: uuid.UUID,
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    job = await _get_job(db, job_id)
    if job.status != "completed" or job.result is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job is {job.status}; results are available once completed",
        )
    return {"id": str(job.id), "config": job.config, **job.result}


@router.get("/history", summary="Recent backtest jobs")
async def backtest_history(
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    jobs = (
        (
            await db.execute(
                select(BacktestJob).order_by(BacktestJob.created_at.desc()).limit(20)
            )
        )
        .scalars()
        .all()
    )
    return [_job_status(j) for j in jobs]
