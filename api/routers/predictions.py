"""Prediction endpoints.

Read the ``predictions`` table populated by the worker's
PredictionPersistenceService (every PredictionReadyEvent lands there), so the
dashboard sees exactly what the models emitted -- including the flat/abstain
calls that never become orders.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, get_db
from api.schemas.predictions import PredictionSchema
from core.models.predictions import Prediction

router = APIRouter(prefix="/predictions")


def _to_schema(row: Prediction) -> PredictionSchema:
    # Explicit mapping: the schema exposes `timestamp`, the table stores
    # `predicted_at` (model-emission time, not row-insertion time).
    return PredictionSchema(
        id=row.id,
        symbol=row.symbol,
        direction=row.direction,  # type: ignore[arg-type]  # SignalDirection value stored as its string
        confidence=row.confidence,
        expected_return=row.expected_return,
        model_id=row.model_id,
        timestamp=row.predicted_at,
    )


async def _query_predictions(
    db: AsyncSession, symbol: str | None, limit: int
) -> list[PredictionSchema]:
    stmt = select(Prediction).order_by(Prediction.predicted_at.desc()).limit(limit)
    if symbol:
        stmt = stmt.where(Prediction.symbol == symbol)
    rows = (await db.execute(stmt)).scalars().all()
    return [_to_schema(row) for row in rows]


@router.get(
    "/latest",
    response_model=list[PredictionSchema],
    summary="Latest predictions",
)
async def get_latest_predictions(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PredictionSchema]:
    """Return the most recent predictions, optionally filtered by symbol."""
    return await _query_predictions(db, symbol, limit)


@router.get(
    "/history",
    response_model=list[PredictionSchema],
    summary="Historical predictions",
)
async def get_prediction_history(
    symbol: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[PredictionSchema]:
    """Return historical predictions with optional filtering."""
    return await _query_predictions(db, symbol, limit)


@router.get(
    "/{prediction_id}",
    response_model=PredictionSchema,
    summary="Prediction detail",
)
async def get_prediction(
    prediction_id: uuid.UUID,
    _user: dict[str, Any] = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PredictionSchema:
    """Return a single prediction by ID."""
    row = (
        await db.execute(select(Prediction).where(Prediction.id == prediction_id))
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found",
        )
    return _to_schema(row)
